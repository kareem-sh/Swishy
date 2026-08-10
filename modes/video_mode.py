import cv2
import logging
from pathlib import Path
from time import perf_counter

from utils.quiet import quiet_native_stderr  # must precede mediapipe

from mediapipe.tasks.python import vision

from config.settings import DEFAULT_FPS, WINDOW_NAME
from feedback.console import print_shot_summary, session_report_to_dict
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.config_loader import load_yaml
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame

log = logging.getLogger(__name__)


def _video_wait_ms(fps: float) -> int:
    cfg = load_yaml("display.yaml")
    speed = float(cfg.get("video_playback_speed", 1.0))
    if not cfg.get("video_playback_sync", True) or speed <= 0:
        return 1
    effective_fps = fps * speed
    return max(1, int(1000 / effective_fps)) if effective_fps > 0 else 33


def _draw_progress_overlay(frame, frame_index: int, total_frames: int) -> None:
    """Show how far through the clip the analysis is.

    This replaced a "Source 30 FPS | Processing 12 FPS | 41% realtime" readout.
    That number was honest but alarming: it is wall-clock THROUGHPUT, and
    "41% realtime" reads as "59% of my video was skipped". Nothing is skipped
    -- every frame is read and analysed, and timestamps come from the frame
    index and the source rate, never from the clock. Frames completed out of
    frames total says what someone waiting actually wants to know.
    """
    if total_frames > 0:
        percent = min(frame_index / total_frames, 1.0) * 100.0
        text = f"Analysing  {frame_index}/{total_frames}  ({percent:.0f}%)"
    else:
        text = f"Analysing  frame {frame_index}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.52, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, 68
    cv2.rectangle(
        frame,
        (x - 5, y - text_h - 7),
        (x + text_w + 5, y + baseline + 5),
        (18, 18, 22),
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, (215, 215, 220),
                thickness, cv2.LINE_AA)


def run_video_mode(video_path: str) -> dict | None:
    with quiet_native_stderr():
        detector = PoseDetector(running_mode=vision.RunningMode.VIDEO)
    pipeline = ShotAnalysisPipeline()
    recorder = SessionRecorder(source_type="video", source_name=Path(video_path).name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipeline.set_fps(fps)
    recorder.fps = fps
    wait_ms = _video_wait_ms(fps)
    frame_index = 0
    processed_frames = 0
    processing_seconds_total = 0.0
    # Shot summaries are collected and printed after the run, so the report is
    # never interleaved with per-frame output.
    completed_shots = []

    while cap.isOpened():
        processing_started = perf_counter()
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = frame_timestamp_ms(frame_index, fps)
        with quiet_native_stderr():
            result = detector.detect_video_frame(rgb_frame, timestamp_ms)
        h, w, _ = frame.shape
        frame_result = pipeline.process_frame(
            result, w, h, timestamp_ms, bgr_frame=frame
        )

        if frame_result.shot_summary:
            completed_shots.append(frame_result.shot_summary)

        annotated = render_frame(rgb_frame, result, frame_result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        recorder.on_frame(frame_index, annotated_bgr, frame_result)

        processing_seconds_total += max(perf_counter() - processing_started, 1e-9)
        processed_frames += 1
        _draw_progress_overlay(annotated_bgr, frame_index + 1, total_frames)

        cv2.imshow(WINDOW_NAME, annotated_bgr)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()

    # The report comes after analysis, in one block, never interleaved.
    for summary in completed_shots:
        print_shot_summary(summary)

    if processed_frames > 0 and processing_seconds_total > 0:
        # Throughput is a developer statistic. It says how long the wait was,
        # not whether the analysis was complete -- every frame was analysed.
        log.info(
            "analysed %d frames in %.1fs (%.1f frames/s)",
            processed_frames,
            processing_seconds_total,
            processed_frames / processing_seconds_total,
        )

    report = recorder.finalize(pipeline=pipeline)
    if report.output_path:
        print(f"\nDetailed report saved: {report.output_path}")
        print(f"  Key frames: {Path(report.output_path).parent / 'frames'}")

    return session_report_to_dict(report)
