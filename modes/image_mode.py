import cv2
from pathlib import Path

from mediapipe.tasks.python import vision

from config.settings import WINDOW_NAME
from feedback.console import print_shot_summary
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from visualization.renderer import render_frame


def run_image_mode(image_path: str):
    detector = PoseDetector(running_mode=vision.RunningMode.IMAGE)
    pipeline = ShotAnalysisPipeline()
    recorder = SessionRecorder(source_type="image", source_name=Path(image_path).name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_image(rgb_image)

    h, w, _ = image.shape
    frame_result = pipeline.process_frame(result, w, h, timestamp_ms=0)

    if frame_result.has_pose:
        print(f"Shooting side: {frame_result.shooting_side}")
        for name, angle in frame_result.angles.items():
            if angle.is_valid:
                print(f"  {name}: {angle.degrees:.1f}° (stable={angle.is_stable})")

    annotated = render_frame(rgb_image, result, frame_result)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    recorder.on_single_image(frame_result, annotated_bgr)

    cv2.imshow(WINDOW_NAME, annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    report = recorder.finalize()
    if report.shots:
        print_shot_summary(report.shots[0].summary)
    if report.output_path:
        print(f"\nDetailed report saved: {report.output_path}")
        print(f"  Key frames: {Path(report.output_path).parent / 'frames'}")
