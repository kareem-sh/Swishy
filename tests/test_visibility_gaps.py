"""Tests for visibility gap tracking in reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angles.calculator import AngleResult
from feedback.visibility_gaps import VisibilityGapTracker
from pipeline import FrameResult


def _frame(
    frame_index: int,
    phase: str = "release",
    elbow_valid: bool = True,
    shot_in_progress: bool = True,
) -> FrameResult:
    return FrameResult(
        has_pose=True,
        phase=phase,
        phase_label=phase.replace("_", " ").title(),
        timestamp_ms=frame_index * 33,
        shooting_side="right",
        shot_in_progress=shot_in_progress,
        angles={
            "right_elbow": AngleResult("right_elbow", 160.0, elbow_valid, elbow_valid),
            "right_knee": AngleResult("right_knee", 110.0, True, True),
            "right_index_align": AngleResult("right_index_align", 165.0, True, True),
            "right_shoulder": AngleResult("right_shoulder", 90.0, True, True),
        },
        world_landmarks={
            "right_wrist": {"is_reliable": True},
        },
    )


def test_visibility_gap_recorded_after_hold_window():
    tracker = VisibilityGapTracker(hold_frames=3)
    for i in range(8):
        tracker.update(i, _frame(i, elbow_valid=False))

    gaps = tracker.finalize()
    assert len(gaps) == 1
    assert "elbow" in gaps[0].body_part
    assert gaps[0].hold_frames_exceeded > 3
    assert "Release" in gaps[0].summary()


def test_no_gap_when_elbow_visible():
    tracker = VisibilityGapTracker(hold_frames=3)
    for i in range(10):
        tracker.update(i, _frame(i, elbow_valid=True))

    gaps = tracker.finalize()
    assert gaps == []


if __name__ == "__main__":
    test_visibility_gap_recorded_after_hold_window()
    test_no_gap_when_elbow_visible()
    print("Visibility gap tests passed.")
