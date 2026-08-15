"""Headless video analysis for server/backend use (no OpenCV windows)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
from mediapipe.tasks.python import vision

from config.settings import DEFAULT_FPS
from feedback.console import session_report_to_dict
from feedback.session_recorder import SessionRecorder
from pipeline import ShotAnalysisPipeline
from player.profile import build_player_profile
from pose.detector import PoseDetector
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame


def analyze_video(
    video_path: str,
    *,
    height_cm: Optional[float] = None,
    rim_height_m: Optional[float] = None,
    shot_xy_m: Optional[Sequence[float]] = None,
    save_key_frames: bool = True,
    key_frames_dir: Optional[str | Path] = None,
    save_report: bool = False,
) -> dict:
    """
    Run full shot analysis on a video file without opening a GUI window.

    Returns the same JSON-compatible dict as session_report_to_dict().
    """
    video_path = str(video_path)
    detector = PoseDetector(running_mode=vision.RunningMode.VIDEO)
    profile = build_player_profile(height_cm=height_cm)
    pipeline = ShotAnalysisPipeline(
        player=profile,
        rim_height_m=rim_height_m,
        shot_xy_m=tuple(shot_xy_m) if shot_xy_m is not None else None,
    )
    recorder = SessionRecorder(
        source_type="video",
        source_name=Path(video_path).name,
    )
    if not save_report:
        recorder._auto_save_report = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    pipeline.set_fps(fps)
    recorder.fps = fps
    frame_index = 0

    while cap.isOpened():
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

        annotated = render_frame(rgb_frame, result, frame_result)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        recorder.on_frame(frame_index, annotated_bgr, frame_result)
        frame_index += 1

    cap.release()

    report = recorder.finalize(pipeline=pipeline)
    payload = session_report_to_dict(report)

    if save_key_frames and key_frames_dir is not None:
        frames_dir = Path(key_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for shot_dict, detailed in zip(payload["shots"], report.shots):
            saved = _save_first_key_frame(detailed, frames_dir, shot_dict["shot_number"])
            if saved:
                shot_dict["key_frame_path"] = str(saved)

    return payload


def _save_first_key_frame(detailed, frames_dir: Path, shot_number: int) -> Optional[Path]:
    """Save the first available key frame image for a shot."""
    for key_frame in detailed.key_frames:
        frame_index = key_frame.frame_index
        image = detailed.frame_images.get(frame_index)
        if image is None:
            continue
        output_path = frames_dir / f"shot_{shot_number:02d}_{frame_index:05d}.jpg"
        cv2.imwrite(str(output_path), image)
        return output_path
    return None
