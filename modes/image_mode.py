import cv2

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image


def run_image_mode(image_path):

    detector = PoseDetector(
        running_mode=vision.RunningMode.IMAGE
    )

    image = cv2.imread(image_path)

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = detector.detect_image(rgb_image)

    h, w, _ = image.shape

    landmarks = extract_landmarks(result, w, h)

    print(landmarks)

    annotated_image = draw_landmarks_on_image(
        rgb_image,
        result,
        landmarks
    )

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow("Image Mode", annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
