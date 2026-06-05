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

## How It Works
The webcam captures video frames in real time.

MediaPipe detects and tracks the user's hand landmarks.

The app identifies finger positions and gestures, then uses OpenCV to draw on a digital canvas based on the movement of the index finger.

Different modes allow the user to draw, erase, change colors, fill shapes, and interact with the whiteboard.

## Why I Built This

I built this project to practice computer vision, hand gesture recognition, and real-time user interaction.

The goal was to create a fun and visual AI project that shows how computer vision can turn simple hand gestures into an interactive tool.

This project helped me better understand:

- Real-time video processing
- Hand landmark detection
- Gesture-based controls
- Drawing logic with OpenCV
- Building interactive AI portfolio projects

## Requirements

Make sure your computer has:

- Python installed
- A working webcam
- Camera permissions enabled
- Required Python packages installed

## Future Improvements

Some features I would like to add next:

- Gesture-based undo and redo
- Save drawings as image files
- Better shape detection
- Text recognition
- Smoother drawing
- Improved UI
- More drawing tools
- Presentation mode
- Project Status

## Author 
Built by Nargess Hassani

This project demonstrates real-time computer vision, hand gesture recognition, user interaction design, and basic shape recognition using Python, OpenCV, and MediaPipe.
