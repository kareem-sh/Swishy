"""Tests for moving-camera rim smoothing and correction gates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import RimDetection
from ball.rim_tracker import RimMotionSmoother, is_plausible_rim_correction


def _rim(x: float, y: float, width: float, frame: int = 0) -> RimDetection:
    height = width * 0.5
    return RimDetection(
        center_xy=(x, y),
        bbox_xyxy=(x - width / 2, y - height / 2, x + width / 2, y + height / 2),
        confidence=0.9,
        frame_index=frame,
        timestamp_ms=frame * 33,
    )


def test_smoother_follows_camera_motion() -> None:
    smoother = RimMotionSmoother(position_alpha=0.75, size_alpha=0.5)
    first = smoother.update(_rim(100, 100, 80))
    moved = smoother.update(_rim(120, 110, 84, frame=1))

    assert first.center_xy == (100.0, 100.0)
    assert moved.center_xy == (115.0, 107.5)
    assert moved.bbox_xyxy[2] - moved.bbox_xyxy[0] == 82.0


def test_oversized_yolo_box_is_rejected() -> None:
    tracked = _rim(500, 300, 100)
    oversized = _rim(510, 305, 190)

    assert not is_plausible_rim_correction(
        oversized,
        tracked,
        maximum_center_distance_px=120,
        maximum_center_distance_scale=1.5,
        maximum_width_change_ratio=1.5,
    )


def test_normal_camera_movement_is_accepted() -> None:
    tracked = _rim(500, 300, 100)
    corrected = _rim(525, 315, 110)

    assert is_plausible_rim_correction(
        corrected,
        tracked,
        maximum_center_distance_px=120,
        maximum_center_distance_scale=1.5,
        maximum_width_change_ratio=1.5,
    )


if __name__ == "__main__":
    test_smoother_follows_camera_motion()
    test_oversized_yolo_box_is_rejected()
    test_normal_camera_movement_is_accepted()
    print("All moving-camera rim tracking tests passed.")
