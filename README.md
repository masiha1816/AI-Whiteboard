# AI Whiteboard

AI Whiteboard is a real-time computer vision application that lets users draw in the air using hand gestures. The app uses a webcam to track the user's hand, detects pinch gestures for drawing, recognizes rough geometric shapes, and snaps them into clean shapes.

## Features

- Real-time hand tracking using MediaPipe
- Pinch-to-draw interaction
- Smooth air drawing
- Shape recognition
- Shape snapping for circles, squares, rectangles, and triangles
- Fill mode
- Eraser mode
- Multiple drawing colors
- Undo support
- Save drawing as PNG

## Controls

| Key | Action               |
| --- | -------------------- |
| 1   | Green                |
| 2   | Red                  |
| 3   | Blue                 |
| 4   | Yellow               |
| F   | Toggle fill mode     |
| E   | Toggle eraser mode   |
| S   | Snap/recognize shape |
| P   | Save drawing         |
| Z   | Undo                 |
| C   | Clear canvas         |
| Q   | Quit                 |

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy

## How to Run

1. Install dependencies:

```bash
pip3 install -r requirements.txt
```

This project demonstrates real-time computer vision, hand gesture recognition, user interaction design, and basic shape recognition using Python, OpenCV, and MediaPipe.
