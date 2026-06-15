import cv2
import numpy as np
from enum import Enum
from PIL import Image

class Colors(Enum):
    RED = [0, 0, 204]
    ORANGE = [0, 165, 255]
    TOXIC_GREEN = [102, 255, 102]
    YELLOW = [0, 255, 255]

def get_limits(color):
    c = np.uint8([[color]])

    hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = int(hsv_c[0][0][0])

    lower_hue = hue - 10
    upper_hue = hue + 10

    if lower_hue < 0:
        return [
            (np.array([0, 100, 100], dtype=np.uint8), np.array([upper_hue, 255, 255], dtype=np.uint8)),
            (np.array([180 + lower_hue, 100, 100], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
        ]
    if upper_hue > 179:
        return [
            (np.array([lower_hue, 100, 100], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
            (np.array([0, 100, 100], dtype=np.uint8), np.array([upper_hue - 180, 255, 255], dtype=np.uint8)),
        ]
    return [
        (np.array([lower_hue, 100, 100], dtype=np.uint8), np.array([upper_hue, 255, 255], dtype=np.uint8)),
    ]


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: could not open camera")
        return

    hue_ranges = get_limits(Colors.TOXIC_GREEN.value)

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        mirrored_frame = cv2.flip(frame, 1)

        hsv_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2HSV)

        mask = None
        for lower_limit, upper_limit in hue_ranges:
            range_mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
            mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

        mask_ = Image.fromarray(mask)

        bounding_box = mask_.getbbox()

        if bounding_box is not None:
            x1, y1, x2, y2 = bounding_box

            cv2.rectangle(mirrored_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('webcam', mirrored_frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
