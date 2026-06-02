import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import math
from datetime import datetime

MODEL_PATH = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)

detector = HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(1)

canvas = None
prev_x = None
prev_y = None
smooth_x = None
smooth_y = None

DRAW_START = 40
DRAW_STOP = 55
SMOOTHING = 0.35

drawing = False
detected_shape = "None"
save_message = ""
eraser_mode = False
fill_mode = False

undo_stack = []
MAX_UNDO = 10

current_color_name = "Green"
current_color = (0, 255, 0)

colors = {
    "1": ("Green", (0, 255, 0)),
    "2": ("Red", (0, 0, 255)),
    "3": ("Blue", (255, 0, 0)),
    "4": ("Yellow", (0, 255, 255)),
}


def distance(p1, p2, w, h):
    x1 = int(p1.x * w)
    y1 = int(p1.y * h)

    x2 = int(p2.x * w)
    y2 = int(p2.y * h)

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def save_for_undo(canvas):
    global undo_stack

    if canvas is None:
        return

    undo_stack.append(canvas.copy())

    if len(undo_stack) > MAX_UNDO:
        undo_stack.pop(0)


def undo_canvas(current_canvas):
    global undo_stack

    if len(undo_stack) == 0:
        return current_canvas, "Nothing to undo"

    previous_canvas = undo_stack.pop()
    return previous_canvas, "Undo applied"


def get_shape_info(canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return "No shape", None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < 800:
        return "Too small", None

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown", None

    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    sides = len(approx)

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    circularity = 4 * math.pi * area / (perimeter * perimeter)

    if sides == 3:
        return "Triangle", (x, y, w, h, approx)

    if sides == 4:
        if 0.85 <= aspect_ratio <= 1.15:
            return "Square", (x, y, w, h, approx)
        else:
            return "Rectangle", (x, y, w, h, approx)

    if circularity > 0.65:
        return "Circle", (x, y, w, h, approx)

    return "Unknown", (x, y, w, h, approx)


def replace_with_perfect_shape(canvas, shape_name, shape_info, color, fill_mode):
    if shape_info is None:
        return canvas

    x, y, w, h, approx = shape_info

    new_canvas = np.zeros_like(canvas)

    padding = 10

    x1 = x - padding
    y1 = y - padding
    x2 = x + w + padding
    y2 = y + h + padding

    thickness = -1 if fill_mode else 5

    if shape_name == "Circle":
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2

        cv2.circle(
            new_canvas,
            center,
            radius,
            color,
            thickness
        )

    elif shape_name == "Square":
        size = max(w, h)

        center_x = x + w // 2
        center_y = y + h // 2

        top_left = (
            center_x - size // 2,
            center_y - size // 2
        )

        bottom_right = (
            center_x + size // 2,
            center_y + size // 2
        )

        cv2.rectangle(
            new_canvas,
            top_left,
            bottom_right,
            color,
            thickness
        )

    elif shape_name == "Rectangle":
        cv2.rectangle(
            new_canvas,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

    elif shape_name == "Triangle":
        points = approx.reshape(-1, 2)

        if len(points) == 3:
            if fill_mode:
                cv2.fillPoly(
                    new_canvas,
                    [points],
                    color
                )
            else:
                cv2.polylines(
                    new_canvas,
                    [points],
                    isClosed=True,
                    color=color,
                    thickness=5
                )

    else:
        return canvas

    return new_canvas


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        h, w, _ = frame.shape

        thumb_tip = hand[4]
        index_tip = hand[8]

        raw_x = int(index_tip.x * w)
        raw_y = int(index_tip.y * h)

        if smooth_x is None:
            smooth_x = raw_x
            smooth_y = raw_y

        smooth_x = int(
            (SMOOTHING * raw_x) + ((1 - SMOOTHING) * smooth_x)
        )

        smooth_y = int(
            (SMOOTHING * raw_y) + ((1 - SMOOTHING) * smooth_y)
        )

        pinch_distance = distance(
            thumb_tip,
            index_tip,
            w,
            h
        )

        if not drawing and pinch_distance < DRAW_START:
            save_for_undo(canvas)
            drawing = True
            save_message = ""

        elif drawing and pinch_distance > DRAW_STOP:
            drawing = False
            prev_x = None
            prev_y = None

        if eraser_mode:
            pointer_color = (255, 255, 255)
            status = "ERASER"
        else:
            pointer_color = current_color if drawing else (0, 0, 255)
            status = "DRAWING" if drawing else "MOVE"

        cv2.circle(
            frame,
            (smooth_x, smooth_y),
            14 if eraser_mode else 12,
            pointer_color,
            -1
        )

        cv2.putText(
            frame,
            status,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            pointer_color,
            3
        )

        cv2.putText(
            frame,
            f"Pinch: {int(pinch_distance)}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if drawing:
            if prev_x is not None:
                if eraser_mode:
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (smooth_x, smooth_y),
                        (0, 0, 0),
                        30
                    )
                else:
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (smooth_x, smooth_y),
                        current_color,
                        5
                    )

            prev_x = smooth_x
            prev_y = smooth_y

    else:
        prev_x = None
        prev_y = None
        smooth_x = None
        smooth_y = None
        drawing = False

    combined = cv2.add(frame, canvas)

    cv2.putText(
        combined,
        "Pinch=Draw/Erase | 1 Green | 2 Red | 3 Blue | 4 Yellow",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined,
        "F=Fill | E=Eraser | S=Snap | P=Save | Z=Undo | C=Clear | Q=Quit",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined,
        f"Detected: {detected_shape}",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    mode_text = "ERASER" if eraser_mode else "DRAW"
    fill_text = "ON" if fill_mode else "OFF"

    cv2.putText(
        combined,
        f"Mode: {mode_text}",
        (10, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )

    cv2.putText(
        combined,
        f"Color: {current_color_name}",
        (10, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        current_color,
        2
    )

    cv2.putText(
        combined,
        f"Fill: {fill_text}",
        (10, 300),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )

    cv2.putText(
        combined,
        f"Undo steps: {len(undo_stack)}",
        (10, 340),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    if save_message:
        cv2.putText(
            combined,
            save_message,
            (10, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

    cv2.imshow(
        "AI Whiteboard - Undo + Fill + Colors + Eraser",
        combined
    )

    key = cv2.waitKey(1) & 0xFF

    if key != 255 and chr(key) in colors:
        current_color_name, current_color = colors[chr(key)]
        eraser_mode = False
        save_message = ""
        prev_x = None
        prev_y = None

    if key == ord("f"):
        fill_mode = not fill_mode
        save_message = ""

    if key == ord("e"):
        eraser_mode = not eraser_mode
        prev_x = None
        prev_y = None

    if key == ord("s"):
        save_for_undo(canvas)
        detected_shape, shape_info = get_shape_info(canvas)
        canvas = replace_with_perfect_shape(
            canvas,
            detected_shape,
            shape_info,
            current_color,
            fill_mode
        )
        save_message = ""

    if key == ord("z"):
        canvas, save_message = undo_canvas(canvas)
        prev_x = None
        prev_y = None

    if key == ord("p"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_whiteboard_drawing_{timestamp}.png"

        cv2.imwrite(filename, canvas)

        save_message = f"Saved: {filename}"

        print(save_message)

    if key == ord("c"):
        save_for_undo(canvas)
        canvas = np.zeros_like(frame)
        detected_shape = "None"
        save_message = ""

    if key == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()