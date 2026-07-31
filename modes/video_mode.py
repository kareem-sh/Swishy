import cv2
from pathlib import Path
from time import perf_counter

from mediapipe.tasks.python import vision

from config.settings import DEFAULT_FPS, WINDOW_NAME
from feedback.console import print_shot_summary, session_report_to_dict
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.config_loader import load_yaml
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame


def _video_wait_ms(fps: float) -> int:
    cfg = load_yaml("display.yaml")
    speed = float(cfg.get("video_playback_speed", 1.0))
    if not cfg.get("video_playback_sync", True) or speed <= 0:
        return 1
    effective_fps = fps * speed
    return max(1, int(1000 / effective_fps)) if effective_fps > 0 else 33


def _draw_fps_overlay(frame, source_fps: float, processing_fps: float) -> None:
    """Show source FPS beside the measured analysis throughput."""
    realtime_percent = (
        processing_fps / source_fps * 100.0 if source_fps > 0 else 0.0
    )
    text = (
        f"Source: {source_fps:.1f} FPS  |  Processing: {processing_fps:.1f} FPS"
        f"  |  {realtime_percent:.0f}% realtime"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    x, y = 10, 68
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 7),
        (x + text_w + 5, y + baseline + 5),
        (18, 18, 22),
        -1,
    )
    color = (100, 230, 120) if processing_fps >= source_fps else (0, 180, 255)
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def run_video_mode(video_path: str) -> dict | None:
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
    wait_ms = _video_wait_ms(fps)
    frame_index = 0
    processed_frames = 0
    processing_fps = 0.0
    processing_seconds_total = 0.0

    while cap.isOpened():
        processing_started = perf_counter()
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = frame_timestamp_ms(frame_index, fps)
        result = detector.detect_video_frame(rgb_frame, timestamp_ms)
        h, w, _ = frame.shape
        frame_result = pipeline.process_frame(
            result, w, h, timestamp_ms, bgr_frame=frame
        )

        if frame_result.shot_summary:
            print_shot_summary(frame_result.shot_summary)

        annotated = render_frame(rgb_frame, result, frame_result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        recorder.on_frame(frame_index, annotated_bgr, frame_result)

        processing_seconds = max(perf_counter() - processing_started, 1e-9)
        processing_seconds_total += processing_seconds
        processed_frames += 1
        instantaneous_fps = 1.0 / processing_seconds
        # Smooth frame-to-frame fluctuations while remaining responsive.
        processing_fps = (
            instantaneous_fps
            if processing_fps == 0.0
            else 0.15 * instantaneous_fps + 0.85 * processing_fps
        )
        _draw_fps_overlay(annotated_bgr, fps, processing_fps)

        cv2.imshow(WINDOW_NAME, annotated_bgr)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    if processed_frames > 0 and processing_seconds_total > 0:
        average_processing_fps = processed_frames / processing_seconds_total
        realtime_percent = average_processing_fps / fps * 100.0 if fps > 0 else 0.0
        print(
            f"\nVideo FPS: {fps:.1f} | Average processing FPS: "
            f"{average_processing_fps:.1f} | {realtime_percent:.0f}% realtime"
        )

    report = recorder.finalize(pipeline=pipeline)
    if report.output_path:
        print(f"\nDetailed report saved: {report.output_path}")
        print(f"  Key frames: {Path(report.output_path).parent / 'frames'}")

    return session_report_to_dict(report)
