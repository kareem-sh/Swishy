"""Headless video analysis for server / CPU-only VPS use.

Same offline path as ``python main.py <video>`` (read the whole file, then
``segment_offline``). The only differences:

- never opens an OpenCV window
- never keeps replay landmarks
- forces CPU for MediaPipe pose, YOLO, and NanoTrack

``main.py`` is unchanged: local desktop runs may still use GPU and the replay
window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2

from scripts.coach_report import analyze_video as _analyze_video_run


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
    """Analyse one upload and return the same JSON as ``main.analyze_video``.

    Physical inputs belong to this request. They are passed through memory and
    never written to YAML, which keeps concurrent VPS requests isolated.

    ``shot_xy_m`` uses FIBA half-court coordinates ``[x_m, y_m]``. When
    ``rim_height_m`` or ``shot_xy_m`` is omitted, the local YAML value remains
    the fallback for development runs.

    ``save_report`` is accepted for CLI compatibility and is unused: the
    offline path does not write a PDF.
    """
    del save_report
    shot_xy = tuple(shot_xy_m) if shot_xy_m is not None else None
    run = _analyze_video_run(
        video_path,
        height_cm=height_cm,
        rim_height_m=rim_height_m,
        shot_xy_m=shot_xy,
        shooting_hand="auto",
        enable_ball=True,
        keep_landmarks=False,
        inference_device="cpu",
    )
    payload = run.to_payload()

    if save_key_frames and key_frames_dir is not None:
        _extract_key_frames(str(video_path), payload, Path(key_frames_dir))

    return payload


def _extract_key_frames(video_path: str, payload: dict, frames_dir: Path) -> None:
    """Save a still at each shot's start time. No overlay, no window."""
    shots = payload.get("shots") or []
    if not shots:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for shot in shots:
            start_ms = shot.get("start_timestamp_ms")
            if start_ms is None:
                continue
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))
            ok, frame = cap.read()
            if not ok:
                continue
            shot_number = int(shot.get("shot_number") or 0)
            output_path = frames_dir / f"shot_{shot_number:02d}.jpg"
            if cv2.imwrite(str(output_path), frame):
                shot["key_frame_path"] = str(output_path)
    finally:
        cap.release()
