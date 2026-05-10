import cv2
import time

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image

from config.settings import WINDOW_NAME


# ==========================================
# GLOBAL RESULT VARIABLE
# ==========================================

latest_result = None


# ==========================================
# CALLBACK FUNCTION
# ==========================================

def save_result(result, output_image, timestamp_ms):

    global latest_result

    latest_result = result


# ==========================================
# LIVE STREAM MODE
# ==========================================

def run_live_stream():

    global latest_result

    detector = PoseDetector(
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=save_result
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Cannot open webcam")
        return

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print("Ignoring empty frame.")
            continue

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Timestamp required
        timestamp_ms = int(time.time() * 1000)

        # Async detection
        detector.detect_async(
            rgb_frame,
            timestamp_ms
        )

        # Wait for first result
        if latest_result is None:

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        h, w, _ = frame.shape

        # Extract landmarks
        landmarks = extract_landmarks(
            latest_result,
            w,
            h
        )

        # Print coordinates
        if landmarks:

            print("\n====================")
            print("POSE COORDINATES")
            print("====================")

            for name, point in landmarks.items():

                print(
                    f"{name}: "
                    f"x={point['x']} "
                    f"y={point['y']} "
                    f"z={point['z']:.4f}"
                )

        # Draw skeleton
        annotated_image = draw_landmarks_on_image(
            rgb_frame,
            latest_result,
            landmarks
        )

        # RGB -> BGR
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )

        # Show result
        cv2.imshow(
            WINDOW_NAME,
            annotated_image
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()
