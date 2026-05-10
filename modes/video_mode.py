import cv2
import time

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image


def run_video_mode(video_path):

    # Create detector using VIDEO mode
    detector = PoseDetector(
        running_mode=vision.RunningMode.VIDEO
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened correctly
    if not cap.isOpened():
        print("Error opening video.")
        return

    while cap.isOpened():

        # Read frame
        success, frame = cap.read()

        # Stop when video ends
        if not success:
            print("Video finished.")
            break

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Timestamp required by MediaPipe VIDEO mode
        timestamp_ms = int(time.time() * 1000)

        # Run pose detection
        result = detector.detect_video_frame(
            rgb_frame,
            timestamp_ms
        )

        # Get frame size
        h, w, _ = frame.shape

        # Extract landmarks
        landmarks = extract_landmarks(
            result,
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

        # Draw skeleton + angles
        annotated_image = draw_landmarks_on_image(
            rgb_frame,
            result,
            landmarks
        )

        # Convert RGB -> BGR for OpenCV display
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )

        # Show video
        cv2.imshow(
            "Basketball Pose Detection",
            annotated_image
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
