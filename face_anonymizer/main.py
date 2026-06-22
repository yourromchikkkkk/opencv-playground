import os
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELS_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def main():
    model_path = os.path.join(MODELS_DIR, 'blaze_face_short_range.tflite')
    cap = cv2.VideoCapture(0)

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
    

    if not cap.isOpened():
        print("Error: could not open camera")
        return
    
    detector = FaceDetector.create_from_options(options)

    frame_timestamp_ms = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        mirrored_frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        frame_timestamp_ms += 33

        h, w, _ = mirrored_frame.shape

        for detection in result.detections:
            bbox = detection.bounding_box

            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            bw = bbox.width
            bh = bbox.height

            roi = mirrored_frame[y:y+bh, x:x+bw]

            if roi.size:
                mirrored_frame[y:y+bh, x:x+bw] = cv2.GaussianBlur(
                    roi,
                    (99, 99),
                    30
                )


        cv2.imshow('webcam', mirrored_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == '__main__':
    main()