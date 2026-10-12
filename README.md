# Hand Tracker Control

Control your Windows desktop with webcam hand gestures. The app uses MediaPipe hand landmarks and OpenCV to map gestures to mouse movement, clicks, system volume, media play/pause, and app switching.

## Features

- Left hand index finger moves the mouse cursor.
- Left hand thumb/index pinch performs a mouse click.
- Right hand thumb/index distance controls system volume.
- Hold the volume gesture still to lock the current volume.
- Right hand fist toggles play/pause.
- Two-finger left/right swipe switches between open apps.
- On-screen HUD shows active gestures and current mode.
- Command-line options for camera index, confidence, smoothing, and gesture thresholds.

## Requirements

- Windows 10 or newer
- Python 3.10+
- Webcam
- The dependencies in `requirements.txt`

The volume control feature uses Windows audio APIs through `pycaw`, so Linux and macOS are not currently supported.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The MediaPipe model file is included in this repository as `hand_landmarker.task`. If it is missing, the app downloads it automatically on first run.

## Usage

```bash
python main.py
```

Press `Q` while the camera window is focused to quit.

You can also run the compatibility entry point:

```bash
python agent.py
```

## Gesture Guide

| Gesture | Action |
| --- | --- |
| Left index finger | Move mouse |
| Left thumb + index pinch | Click |
| Right thumb/index spread | Adjust volume |
| Hold right volume gesture still | Lock volume |
| Right fist | Play/pause media |
| Two fingers up, swipe left | Previous app |
| Two fingers up, swipe right | Next app |

Because the camera preview is mirrored, the app internally flips MediaPipe handedness so the on-screen left/right labels feel natural.

## Configuration

```bash
python main.py --camera 1 --confidence 0.6 --smooth 6 --click-distance 30
```

Useful options:

- `--camera`: camera index, default `0`
- `--confidence`: MediaPipe detection/tracking confidence, default `0.5`
- `--smooth`: mouse smoothing factor, default `5`
- `--click-distance`: pinch distance in pixels for clicking, default `35`
- `--swipe-threshold`: normalized wrist movement needed for app switching, default `0.25`
- `--keep-failsafe`: keep PyAutoGUI's corner failsafe enabled

Run `python main.py --help` to see every option.

## Troubleshooting

- If the camera does not open, try `python main.py --camera 1` or another camera index.
- If gestures feel too sensitive, increase `--smooth`, `--click-distance`, or `--swipe-threshold`.
- If volume control fails, make sure you are running on Windows with a normal output audio device selected.
- If MediaPipe installation fails, upgrade packaging tools with `python -m pip install --upgrade pip setuptools wheel`.

## Project Structure

```text
.
+-- main.py                # Main hand-control application
+-- agent.py               # Compatibility entry point
+-- hand_landmarker.task   # MediaPipe hand landmark model
+-- requirements.txt       # Python dependencies
+-- README.md              # GitHub project documentation
```
