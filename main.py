"""Camera-based hand controls for mouse, volume, media, and app switching."""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request
from ctypes import POINTER, cast
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from comtypes import CLSCTX_ALL
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("hand_landmarker.task")
WINDOW_TITLE = "Hand Control"

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


@dataclass
class ControlConfig:
    camera_index: int
    model_path: Path
    min_confidence: float
    smooth: float
    click_distance: int
    swipe_threshold: float
    swipe_cooldown: float
    grip_cooldown: float
    lock_hold_time: float
    volume_still_threshold: int
    disable_failsafe: bool


@dataclass
class RuntimeState:
    prev_x: float = 0
    prev_y: float = 0
    clicked: bool = False
    swipe_start_x: float | None = None
    last_swipe_time: float = 0
    swipe_label: str = ""
    swipe_label_timer: float = 0
    volume_locked: bool = False
    locked_vol_db: float | None = None
    vol_lock_start: float | None = None
    vol_prev_dist: float | None = None
    last_grip_time: float = 0
    grip_label: str = ""
    grip_label_timer: float = 0


def parse_args() -> ControlConfig:
    parser = argparse.ArgumentParser(
        description="Control mouse, system volume, media playback, and app switching with hand gestures."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index to open. Default: 0.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe hand landmarker task file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection, presence, and tracking confidence from 0.0 to 1.0.",
    )
    parser.add_argument("--smooth", type=float, default=5.0, help="Mouse smoothing factor.")
    parser.add_argument("--click-distance", type=int, default=35, help="Pinch distance for click.")
    parser.add_argument("--swipe-threshold", type=float, default=0.25, help="Normalized wrist movement for swipe.")
    parser.add_argument("--swipe-cooldown", type=float, default=1.0, help="Seconds between app switch swipes.")
    parser.add_argument("--grip-cooldown", type=float, default=1.0, help="Seconds between play/pause gestures.")
    parser.add_argument("--lock-hold-time", type=float, default=1.5, help="Seconds to hold volume still before lock.")
    parser.add_argument(
        "--volume-still-threshold",
        type=int,
        default=15,
        help="Pixel movement allowed while locking volume.",
    )
    parser.add_argument(
        "--keep-failsafe",
        action="store_true",
        help="Keep PyAutoGUI's corner failsafe enabled.",
    )
    args = parser.parse_args()

    confidence = min(1.0, max(0.0, args.confidence))
    return ControlConfig(
        camera_index=args.camera,
        model_path=args.model,
        min_confidence=confidence,
        smooth=max(1.0, args.smooth),
        click_distance=max(1, args.click_distance),
        swipe_threshold=max(0.01, args.swipe_threshold),
        swipe_cooldown=max(0.0, args.swipe_cooldown),
        grip_cooldown=max(0.0, args.grip_cooldown),
        lock_hold_time=max(0.1, args.lock_hold_time),
        volume_still_threshold=max(1, args.volume_still_threshold),
        disable_failsafe=not args.keep_failsafe,
    )


def ensure_model(model_path: Path) -> None:
    if model_path.exists():
        return

    print(f"Downloading MediaPipe hand model to {model_path}...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Model download complete.")


def create_landmarker(config: ControlConfig) -> vision.HandLandmarker:
    ensure_model(config.model_path)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.model_path)),
        num_hands=2,
        min_hand_detection_confidence=config.min_confidence,
        min_hand_presence_confidence=config.min_confidence,
        min_tracking_confidence=config.min_confidence,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def get_volume_control() -> tuple[IAudioEndpointVolume, float, float]:
    try:
        speakers = AudioUtilities.GetSpeakers()
        interface = speakers._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        min_vol, max_vol = volume.GetVolumeRange()[:2]
        return volume, min_vol, max_vol
    except Exception as exc:
        raise RuntimeError(
            "Could not access the Windows audio endpoint. This project currently requires Windows audio APIs."
        ) from exc


def count_extended_fingers(landmarks) -> list[int]:
    """Return extended finger indices: 0=thumb, 1=index, 2=middle, 3=ring, 4=pinky."""
    extended = []
    if landmarks[4].x < landmarks[3].x:
        extended.append(0)

    finger_pairs = [(8, 6, 1), (12, 10, 2), (16, 14, 3), (20, 18, 4)]
    for tip, pip, finger_index in finger_pairs:
        if landmarks[tip].y < landmarks[pip].y:
            extended.append(finger_index)
    return extended


def is_grip(landmarks) -> bool:
    extended = count_extended_fingers(landmarks)
    return all(finger not in extended for finger in [1, 2, 3, 4])


def draw_landmarks(frame, hand_landmarks, color=(0, 255, 0)) -> None:
    height, width, _ = frame.shape
    for start, end in HAND_CONNECTIONS:
        x0 = int(hand_landmarks[start].x * width)
        y0 = int(hand_landmarks[start].y * height)
        x1 = int(hand_landmarks[end].x * width)
        y1 = int(hand_landmarks[end].y * height)
        cv2.line(frame, (x0, y0), (x1, y1), (200, 200, 200), 1)

    for landmark in hand_landmarks:
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(frame, (cx, cy), 4, color, cv2.FILLED)


def check_swipe(current_x: float, state: RuntimeState, config: ControlConfig) -> str | None:
    now = time.monotonic()
    if now - state.last_swipe_time < config.swipe_cooldown:
        state.swipe_start_x = current_x
        return None

    if state.swipe_start_x is None:
        state.swipe_start_x = current_x
        return None

    delta = current_x - state.swipe_start_x
    if abs(delta) < config.swipe_threshold:
        return None

    state.swipe_start_x = current_x
    state.last_swipe_time = now
    return "right" if delta > 0 else "left"


def draw_hud(frame) -> None:
    height, width, _ = frame.shape
    hud_lines = [
        "L-hand: index finger = mouse",
        "L-hand: pinch        = click",
        "R-hand: spread       = volume",
        "R-hand: hold still   = lock vol",
        "R-hand: fist/grip    = play/pause",
        "2-finger swipe L/R   = switch app",
        "Q = quit",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(
            frame,
            line,
            (width - 360, 30 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def split_hands(results) -> tuple[object | None, object | None]:
    left_hand = None
    right_hand = None
    if not results.hand_landmarks or not results.handedness:
        return left_hand, right_hand

    for index, handedness in enumerate(results.handedness):
        label = handedness[0].category_name
        landmarks = results.hand_landmarks[index]

        # The frame is mirrored for a natural webcam view, so reported handedness is reversed.
        if label == "Right":
            left_hand = landmarks
        else:
            right_hand = landmarks

    return left_hand, right_hand


def handle_left_hand(frame, hand, state: RuntimeState, config: ControlConfig, screen_size: tuple[int, int]) -> None:
    height, width, _ = frame.shape
    screen_w, screen_h = screen_size
    draw_landmarks(frame, hand, color=(0, 255, 0))

    thumb = (int(hand[4].x * width), int(hand[4].y * height))
    index = (int(hand[8].x * width), int(hand[8].y * height))
    cv2.putText(frame, "LEFT: MOUSE", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    target_x = np.interp(hand[8].x, [0.1, 0.9], [0, screen_w])
    target_y = np.interp(hand[8].y, [0.1, 0.9], [0, screen_h])
    cur_x = state.prev_x + (target_x - state.prev_x) / config.smooth
    cur_y = state.prev_y + (target_y - state.prev_y) / config.smooth
    pyautogui.moveTo(cur_x, cur_y)
    state.prev_x, state.prev_y = cur_x, cur_y

    dist_to_thumb = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
    if dist_to_thumb < config.click_distance:
        if not state.clicked:
            pyautogui.click()
            state.clicked = True
        cv2.circle(frame, index, 15, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "CLICK", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        state.clicked = False
        cv2.circle(frame, index, 12, (0, 255, 0), cv2.FILLED)


def handle_right_hand(
    frame,
    hand,
    state: RuntimeState,
    config: ControlConfig,
    volume,
    min_vol: float,
    max_vol: float,
    left_hand_visible: bool,
) -> None:
    height, width, _ = frame.shape
    draw_landmarks(frame, hand, color=(0, 200, 255))
    thumb = (int(hand[4].x * width), int(hand[4].y * height))
    index = (int(hand[8].x * width), int(hand[8].y * height))
    y_offset = 120 if left_hand_visible else 50
    now = time.monotonic()

    if is_grip(hand):
        if now - state.last_grip_time > config.grip_cooldown:
            pyautogui.press("playpause")
            state.last_grip_time = now
            state.grip_label = "PLAY / PAUSE"
            state.grip_label_timer = now
        cv2.putText(
            frame,
            "RIGHT: GRIP -> PLAY/PAUSE",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 100),
            2,
        )
        return

    cv2.putText(frame, "RIGHT: VOLUME", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    dist = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
    vol_db = float(np.interp(dist, [30, 200], [min_vol, max_vol]))
    vol_pct = int(np.interp(dist, [30, 200], [0, 100]))

    if state.vol_prev_dist is not None and abs(dist - state.vol_prev_dist) < config.volume_still_threshold:
        if state.vol_lock_start is None:
            state.vol_lock_start = now
        elif not state.volume_locked and now - state.vol_lock_start >= config.lock_hold_time:
            state.volume_locked = True
            state.locked_vol_db = vol_db
    else:
        state.vol_lock_start = None
        if state.volume_locked:
            state.volume_locked = False
            state.locked_vol_db = None

    state.vol_prev_dist = dist

    if state.volume_locked and state.locked_vol_db is not None:
        volume.SetMasterVolumeLevel(state.locked_vol_db, None)
        locked_pct = int(np.interp(state.locked_vol_db, [min_vol, max_vol], [0, 100]))
        cv2.putText(
            frame,
            f"VOL LOCKED {locked_pct}%",
            (10, y_offset + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 220, 0),
            2,
        )
        bar_color = (0, 220, 255)
        vol_for_bar = locked_pct
    else:
        volume.SetMasterVolumeLevel(vol_db, None)
        bar_color = (0, 200, 255)
        vol_for_bar = vol_pct
        if state.vol_lock_start is not None:
            progress = min(1.0, (now - state.vol_lock_start) / config.lock_hold_time)
            cv2.putText(
                frame,
                f"Locking... {int(progress * 100)}%",
                (10, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 0),
                2,
            )

    cv2.line(frame, thumb, index, bar_color, 3)
    cv2.circle(frame, thumb, 12, bar_color, cv2.FILLED)
    cv2.circle(frame, index, 12, bar_color, cv2.FILLED)

    bar_x = 50
    bar_top, bar_bot = 150, 400
    bar_y = int(np.interp(vol_for_bar, [0, 100], [bar_bot, bar_top]))
    cv2.rectangle(frame, (bar_x, bar_top), (bar_x + 35, bar_bot), (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 35, bar_bot), bar_color, cv2.FILLED)
    cv2.putText(frame, f"{vol_for_bar}%", (bar_x - 5, bar_bot + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)


def handle_swipe(frame, active_hand, state: RuntimeState, config: ControlConfig) -> None:
    height, width, _ = frame.shape
    if not active_hand:
        state.swipe_start_x = None
        return

    extended = count_extended_fingers(active_hand)
    two_finger_mode = 1 in extended and 2 in extended and 0 not in extended
    if not two_finger_mode:
        state.swipe_start_x = None
        return

    swipe = check_swipe(active_hand[0].x, state, config)
    ix = int(active_hand[8].x * width)
    iy = int(active_hand[8].y * height)
    mx = int(active_hand[12].x * width)
    my = int(active_hand[12].y * height)
    cv2.circle(frame, (ix, iy), 10, (255, 200, 0), cv2.FILLED)
    cv2.circle(frame, (mx, my), 10, (255, 200, 0), cv2.FILLED)
    cv2.putText(frame, "SWIPE MODE", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    if swipe == "left":
        pyautogui.hotkey("alt", "shift", "tab")
        state.swipe_label = "<< PREV APP"
        state.swipe_label_timer = time.monotonic()
    elif swipe == "right":
        pyautogui.hotkey("alt", "tab")
        state.swipe_label = "NEXT APP >>"
        state.swipe_label_timer = time.monotonic()


def draw_temporary_labels(frame, state: RuntimeState) -> None:
    height, width, _ = frame.shape
    now = time.monotonic()

    if state.grip_label and now - state.grip_label_timer < 1.0:
        size = cv2.getTextSize(state.grip_label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        cv2.putText(
            frame,
            state.grip_label,
            ((width - size[0]) // 2, height // 2 - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 100, 100),
            3,
        )
    else:
        state.grip_label = ""

    if state.swipe_label and now - state.swipe_label_timer < 0.8:
        size = cv2.getTextSize(state.swipe_label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        cv2.putText(
            frame,
            state.swipe_label,
            ((width - size[0]) // 2, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 140, 0),
            3,
        )
    else:
        state.swipe_label = ""


def run(config: ControlConfig) -> int:
    pyautogui.FAILSAFE = not config.disable_failsafe
    pyautogui.PAUSE = 0

    landmarker = create_landmarker(config)
    volume, min_vol, max_vol = get_volume_control()
    screen_size = pyautogui.size()
    state = RuntimeState()
    cap = cv2.VideoCapture(config.camera_index)

    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(f"Could not open camera index {config.camera_index}.")

    start_time = time.monotonic()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Camera frame could not be read. Exiting.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ms = int((time.monotonic() - start_time) * 1000)
            results = landmarker.detect_for_video(mp_image, frame_ms)

            left_hand, right_hand = split_hands(results)
            if left_hand:
                handle_left_hand(frame, left_hand, state, config, screen_size)
            if right_hand:
                handle_right_hand(frame, right_hand, state, config, volume, min_vol, max_vol, left_hand is not None)

            active_hand = right_hand or left_hand
            handle_swipe(frame, active_hand, state, config)
            draw_temporary_labels(frame, state)

            if not left_hand and not right_hand:
                cv2.putText(frame, "NO HAND DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

            draw_hud(frame)
            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    return 0


def main() -> int:
    config = parse_args()
    try:
        return run(config)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
