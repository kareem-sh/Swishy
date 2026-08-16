"""Focused tests for ball smoothing, prediction, and rim geometry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.detector import BallDetector
from ball.models import BallDetection, CourtDetections, RimDetection
from ball.tracker import BallTracker


def _detection(x: float, y: float, frame: int, timestamp_ms: int):
    return BallDetection(
        center_xy=(x, y),
        bbox_xyxy=(x - 5, y - 5, x + 5, y + 5),
        confidence=0.9,
        frame_index=frame,
        timestamp_ms=timestamp_ms,
    )


def test_tracker_follows_fast_measurements_without_moving_average_lag():
    tracker = BallTracker(
        max_gap_ms=200,
        position_alpha=0.8,
        velocity_alpha=0.5,
    )
    tracker.update(_detection(0, 0, 0, 0), 0, 0)
    tracker.update(_detection(10, 0, 1, 100), 1, 100)
    third = tracker.update(_detection(20, 0, 2, 200), 2, 200)

    assert third is not None
    assert third.x > 15.0
    assert third.x <= 20.0
    assert third.velocity_xy[0] > 0


def test_tracker_predicts_only_short_missing_gaps():
    tracker = BallTracker(max_gap_ms=200)
    tracker.update(_detection(0, 0, 0, 0), 0, 0)
    tracker.update(_detection(10, 0, 1, 100), 1, 100)

    predicted = tracker.update(None, 2, 200)
    assert predicted is not None
    assert predicted.is_interpolated
    assert predicted.x > 10

    assert tracker.update(None, 4, 400) is None


def test_tracker_labels_detection_and_prediction_provenance():
    tracker = BallTracker(max_gap_ms=200)
    detection = _detection(10, 20, 0, 0)
    detection.measurement_source = "yolo"

    measured = tracker.update(detection, 0, 0)
    predicted = tracker.update(None, 99, 100)

    assert measured is not None
    assert measured.measurement_source == "yolo"
    assert measured.is_direct_observation
    assert predicted is not None
    assert predicted.measurement_source == "predicted"
    assert not predicted.is_visual_observation


def test_distant_rim_jump_is_rejected_after_lock():
    detector = BallDetector.__new__(BallDetector)
    detector.rim_lock_max_shift_ratio = 0.75
    ball = _detection(100, 100, 0, 0)
    detector._sticky_rim_det = RimDetection(
        center_xy=(100, 100),
        bbox_xyxy=(50, 50, 150, 150),
        confidence=0.8,
        frame_index=0,
        timestamp_ms=0,
    )
    distant_rim = RimDetection(
        center_xy=(300, 100),
        bbox_xyxy=(250, 50, 350, 150),
        confidence=0.8,
        frame_index=0,
        timestamp_ms=0,
    )

    result = detector._reject_rim_jump(
        CourtDetections(ball=ball, rim=distant_rim)
    )
    assert result.ball is ball
    assert result.rim is None


if __name__ == "__main__":
    test_tracker_follows_fast_measurements_without_moving_average_lag()
    test_tracker_predicts_only_short_missing_gaps()
    test_tracker_labels_detection_and_prediction_provenance()
    test_distant_rim_jump_is_rejected_after_lock()
    print("All ball tracking tests passed.")
