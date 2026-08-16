"""
Video metadata validation.

Container metadata is not trustworthy. Real examples encountered in testing:
a WebM whose header reported 1000 fps (millisecond timebase), and clips at
12 fps. Feeding either into temporal calculations silently produces nonsense
-- every phase duration collapsed to 0.00 s with no warning.

So frame rate is validated once, up front, and an untrustworthy value is a
structured error rather than a silent fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# A camera producing fewer than 5 fps cannot resolve a release.
#
# The upper bound is the fastest capture rate consumer hardware offers (960 fps
# slow-motion on flagship phones). Anything above it is not a frame rate at all
# but a timebase leaking into the frame-rate field -- 1000 (milliseconds) and
# 90000 (MPEG) are the usual culprits. Bounds are deliberately generous: this
# rejects malformed metadata, it does not express a preference about quality.
MIN_PLAUSIBLE_FPS = 5.0
MAX_PLAUSIBLE_FPS = 960.0

# Above this we accept the clip but warn: high-speed capture is real, and
# slow-motion footage is legitimate input, but frame-count heuristics degrade.
HIGH_FPS_WARNING = 120.0


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    is_valid: bool
    error: Optional[str] = None
    warning: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def validate_fps(raw_fps, path: Path, frame_count: int) -> tuple[float, Optional[str], Optional[str]]:
    """Return (fps, error, warning). `error` non-None means do not proceed."""
    if raw_fps is None:
        return 0.0, f"{path.name}: container reports no frame rate.", None

    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return 0.0, f"{path.name}: frame rate {raw_fps!r} is not a number.", None

    if fps != fps or fps in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0, f"{path.name}: frame rate is not finite ({raw_fps}).", None

    if fps <= 0:
        return 0.0, f"{path.name}: frame rate is {fps:g}, which is not usable.", None

    if fps < MIN_PLAUSIBLE_FPS:
        return fps, (
            f"{path.name}: frame rate {fps:g} fps is below {MIN_PLAUSIBLE_FPS:g} fps. "
            "A shot release cannot be located reliably at this rate."
        ), None

    if fps > MAX_PLAUSIBLE_FPS:
        return fps, (
            f"{path.name}: frame rate {fps:g} fps is implausible. The container is "
            "most likely reporting a timebase rather than a frame rate. "
            "Re-encode the file with a correct frame rate before analysing it."
        ), None

    warning = None
    if fps > HIGH_FPS_WARNING:
        warning = (
            f"{path.name}: {fps:g} fps is high-speed or slow-motion footage. "
            "Analysis runs on timestamps, so this is supported, but wall-clock "
            "durations will not match real time."
        )

    return fps, None, warning


def probe_video(path) -> VideoMetadata:
    """Read and validate a video's metadata without decoding the whole file."""
    import cv2

    path = Path(path)
    if not path.exists():
        return VideoMetadata(path, 0.0, 0, 0, 0, False, f"No such file: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return VideoMetadata(
            path, 0.0, 0, 0, 0, False,
            f"{path.name}: cannot be opened. The file may be corrupt or in an "
            "unsupported format.",
        )

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Consistency check: a container can advertise a frame count it cannot
    # actually deliver, which is the same defect class as a bogus frame rate --
    # both make index-derived timestamps meaningless. Seek near the declared
    # end and confirm a frame is really there.
    decodable = True
    if frame_count > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        decodable = cap.grab()
    cap.release()

    fps, error, warning = validate_fps(raw_fps, path, frame_count)

    if error is None and frame_count <= 0:
        error = f"{path.name}: container reports {frame_count} frames."

    if error is None and not decodable:
        error = (
            f"{path.name}: the container claims {frame_count} frames but the "
            "last one cannot be decoded, so its metadata cannot be trusted for "
            "timing. Re-encode the file before analysing it."
        )

    return VideoMetadata(
        path=path,
        fps=fps,
        frame_count=max(frame_count, 0),
        width=width,
        height=height,
        is_valid=error is None,
        error=error,
        warning=warning,
    )
