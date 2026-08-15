import cv2
from datetime import datetime

from mediapipe.tasks.python import vision

from config.settings import DEFAULT_FPS, WINDOW_NAME
from feedback.console import print_shot_summary
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame

latest_result = None


def _save_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


def run_live_stream():
    detector = PoseDetector(
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=_save_result,
    )
    pipeline = ShotAnalysisPipeline()
    session_name = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    recorder = SessionRecorder(source_type="live", source_name=session_name)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    pipeline.set_fps(fps)
    recorder.fps = fps
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = frame_timestamp_ms(frame_index, fps)

        detector.detect_async(rgb_frame, timestamp_ms)

        if latest_result is None:
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1
            continue

        h, w, _ = frame.shape
        frame_result = pipeline.process_frame(
            latest_result, w, h, timestamp_ms, bgr_frame=frame
        )

        if frame_result.shot_summary:
            print_shot_summary(frame_result.shot_summary)

        annotated = render_frame(rgb_frame, latest_result, frame_result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        recorder.on_frame(frame_index, annotated_bgr, frame_result)

        cv2.imshow(WINDOW_NAME, annotated_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    report = recorder.finalize(pipeline=pipeline)
    if report.output_path:
        from pathlib import Path
        print(f"\nDetailed report saved: {report.output_path}")
        print(f"  Key frames: {Path(report.output_path).parent / 'frames'}")
