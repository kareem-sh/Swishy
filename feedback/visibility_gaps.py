"""Track periods when shooting landmarks/angles were unreliable during a shot."""

from dataclasses import dataclass
from typing import Dict, List

from config.settings import VISIBILITY_HOLD_FRAMES
from feedback.report_models import VisibilityGapNote
from pipeline import FrameResult


TRACKED_BODY_PARTS = {
    "elbow": "shooting elbow",
    "knee": "shooting knee",
    "index_align": "index finger alignment",
    "shoulder": "shooting shoulder",
    "wrist": "shooting wrist",
}


@dataclass
class _OpenGap:
    body_part: str
    phase: str
    phase_label: str
    start_ms: int
    start_frame: int
    invalid_frames: int = 0


class VisibilityGapTracker:
    """Accumulate visibility gaps across frames of one shot attempt."""

    def __init__(self, hold_frames: int = VISIBILITY_HOLD_FRAMES):
        self._hold_frames = hold_frames
        self._open: Dict[str, _OpenGap] = {}
        self._closed: List[VisibilityGapNote] = []
        self._last_ms = 0
        self._last_frame = 0

    def reset(self):
        self._open.clear()
        self._closed.clear()
        self._last_ms = 0
        self._last_frame = 0

    def update(self, frame_index: int, frame_result: FrameResult):
        if not frame_result.has_pose or not frame_result.shot_in_progress:
            return

        self._last_ms = frame_result.timestamp_ms
        self._last_frame = frame_index

        side = frame_result.shooting_side
        phase = frame_result.phase
        phase_label = frame_result.phase_label
        ts = frame_result.timestamp_ms

        checks = {
            "elbow": _angle_valid(frame_result, f"{side}_elbow"),
            "knee": _angle_valid(frame_result, f"{side}_knee"),
            "index_align": _angle_valid(frame_result, f"{side}_index_align"),
            "shoulder": _angle_valid(frame_result, f"{side}_shoulder"),
            "wrist": _landmark_valid(frame_result, f"{side}_wrist"),
        }

        for part, is_valid in checks.items():
            if is_valid:
                self._close_gap(part, ts, frame_index)
            else:
                self._note_invalid(part, phase, phase_label, ts, frame_index)

    def finalize(self) -> List[VisibilityGapNote]:
        for part in list(self._open.keys()):
            self._close_gap(part, self._last_ms, self._last_frame)
        notes = list(self._closed)
        self._closed.clear()
        return notes

    def _note_invalid(self, part: str, phase: str, phase_label: str, ts: int, frame_index: int):
        gap = self._open.get(part)
        if gap is None:
            self._open[part] = _OpenGap(
                body_part=part,
                phase=phase,
                phase_label=phase_label,
                start_ms=ts,
                start_frame=frame_index,
                invalid_frames=1,
            )
            return

        if gap.phase != phase:
            self._close_gap(part, ts, frame_index)
            self._open[part] = _OpenGap(
                body_part=part,
                phase=phase,
                phase_label=phase_label,
                start_ms=ts,
                start_frame=frame_index,
                invalid_frames=1,
            )
            return

        gap.invalid_frames += 1

    def _close_gap(self, part: str, end_ms: int, end_frame: int):
        gap = self._open.pop(part, None)
        if gap is None or gap.invalid_frames <= self._hold_frames:
            return

        self._closed.append(
            VisibilityGapNote(
                body_part=TRACKED_BODY_PARTS.get(part, part),
                phase=gap.phase,
                phase_label=gap.phase_label,
                start_ms=gap.start_ms,
                end_ms=end_ms,
                start_frame=gap.start_frame,
                end_frame=end_frame,
                hold_frames_exceeded=gap.invalid_frames,
            )
        )


def _angle_valid(frame_result: FrameResult, key: str) -> bool:
    angle = frame_result.angles.get(key)
    return angle is not None and angle.is_valid


def _landmark_valid(frame_result: FrameResult, key: str) -> bool:
    if frame_result.world_landmarks is None:
        return False
    lm = frame_result.world_landmarks.get(key)
    return lm is not None and lm.get("is_reliable", False)
