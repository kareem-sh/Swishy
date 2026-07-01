import cv2
from pathlib import Path

from mediapipe.tasks.python import vision

from config.settings import DEFAULT_FPS, WINDOW_NAME
from feedback.console import print_shot_summary
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.config_loader import load_yaml
from utils.performance import get_pose_model_path
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame


def _video_wait_ms(fps: float) -> int:
    cfg = load_yaml("display.yaml")
    speed = float(cfg.get("video_playback_speed", 1.0))
    if not cfg.get("video_playback_sync", True) or speed <= 0:
        return 1
    effective_fps = fps * speed
    return max(1, int(1000 / effective_fps)) if effective_fps > 0 else 33


def run_video_mode(video_path: str):
    detector = PoseDetector(running_mode=vision.RunningMode.VIDEO)
    pipeline = ShotAnalysisPipeline()
    recorder = SessionRecorder(source_type="video", source_name=Path(video_path).name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    pipeline.set_fps(fps)
    recorder.fps = fps
    print(f"Pose backend: {detector.delegate.upper()} | model: {Path(get_pose_model_path()).name}")
    frame_index = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = frame_timestamp_ms(frame_index, fps)

        result = detector.detect_video_frame(rgb_frame, timestamp_ms)
        h, w, _ = frame.shape
        frame_result = pipeline.process_frame(result, w, h, timestamp_ms)

        if frame_result.shot_summary:
            print_shot_summary(frame_result.shot_summary)

        annotated = render_frame(rgb_frame, result, frame_result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        recorder.on_frame(frame_index, annotated_bgr, frame_result)

        cv2.imshow(WINDOW_NAME, annotated_bgr)
        if cv2.waitKey(_video_wait_ms(fps)) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    report = recorder.finalize(pipeline=pipeline)
    if report.output_path:
        print(f"\nDetailed report saved: {report.output_path}")
        print(f"  Key frames: {Path(report.output_path).parent / 'frames'}")
