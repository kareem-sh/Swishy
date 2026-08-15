"""Synthetic tests for release, rim crossing, and shot completion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallDetection, BallSnapshot, RimDetection, ShotOutcome
from ball.shot_state_machine import (
    BallShotState,
    BallShotStateMachine,
    BallTrackingStatus,
)
from feedback.console import shot_summary_to_dict
from feedback.release_timing import release_timing_fields
from feedback.shot_tracker import ShotTracker
from utils.frame_buffer import FrameBuffer, FrameSnapshot


def _ball(
    x: float,
    y: float,
    frame: int,
    timestamp_ms: int,
    velocity=(0.0, 0.0),
    observed: bool = True,
):
    detection = None
    if observed:
        detection = BallDetection(
            center_xy=(x, y),
            bbox_xyxy=(x - 10, y - 10, x + 10, y + 10),
            confidence=0.9,
            frame_index=frame,
            timestamp_ms=timestamp_ms,
        )
    snapshot = BallSnapshot(
        timestamp_ms=timestamp_ms,
        frame_index=frame,
        center_xy=(x, y),
        confidence=0.9 if observed else 0.3,
        velocity_xy=velocity,
        is_interpolated=not observed,
    )
    return detection, snapshot


def _rim(frame: int = 0, timestamp_ms: int = 0) -> RimDetection:
    return RimDetection(
        center_xy=(500.0, 300.0),
        bbox_xyxy=(450.0, 300.0, 550.0, 430.0),
        confidence=0.9,
        frame_index=frame,
        timestamp_ms=timestamp_ms,
    )


def _update(
    machine: BallShotStateMachine,
    x: float,
    y: float,
    frame: int,
    timestamp_ms: int,
    velocity,
    pose_phase=None,
    wrist_xy=None,
    observed=True,
    ankle_y=None,
):
    detection, snapshot = _ball(
        x,
        y,
        frame,
        timestamp_ms,
        velocity,
        observed,
    )
    return machine.update(
        ball_detection=detection,
        ball_snapshot=snapshot,
        rim_detection=_rim(frame, timestamp_ms),
        wrist_xy=wrist_xy,
        ankle_y=ankle_y,
        pose_phase=pose_phase,
        timestamp_ms=timestamp_ms,
    )


# Where the feet stand, in image pixels. Image y grows downward, so the floor
# has a LARGER y than the ball. `ankle_release_fallback_px` is 300, so a ball at
# y=400 is confirmed released against an ankle at y=700 and not before.
_STANDING_ANKLE_Y = 700.0


def _release(machine: BallShotStateMachine) -> None:
    _update(
        machine,
        100,
        500,
        0,
        0,
        (0, 0),
        pose_phase="rise",
        wrist_xy=(100, 500),
        ankle_y=_STANDING_ANKLE_Y,
    )
    result = _update(
        machine,
        200,
        400,
        1,
        100,
        (500, -500),
        pose_phase="release",
        wrist_xy=(100, 500),
        ankle_y=_STANDING_ANKLE_Y,
    )
    assert result.state == BallShotState.ASCENDING


def test_clean_inside_crossing_becomes_made() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500, 240, 2, 200, (100, -200))
    _update(machine, 500, 280, 3, 300, (0, 200))
    crossing = _update(machine, 500, 315, 4, 400, (0, 350))
    made = _update(machine, 502, 335, 5, 500, (20, 200))

    assert crossing.crossed_rim_this_frame
    assert made.state == BallShotState.MADE
    assert made.outcome is not None
    assert made.outcome.result == "made"
    assert made.outcome.entry_frame == 4


def test_outside_crossing_becomes_missed() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 560, 250, 2, 200, (100, -150))
    _update(machine, 560, 280, 3, 300, (0, 200))
    _update(machine, 562, 315, 4, 400, (20, 350))
    missed = _update(machine, 565, 340, 5, 500, (30, 250))

    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None
    assert missed.outcome.result == "missed"


def test_miss_confirmation_uses_elapsed_time_not_frame_count() -> None:
    machine = BallShotStateMachine()
    machine.miss_confirmation_s = 0.05
    _release(machine)

    _update(machine, 570, 250, 2, 200, (100, -150))
    _update(machine, 570, 280, 3, 300, (0, 200))
    crossing = _update(machine, 570, 310, 4, 400, (0, 300))
    first = _update(machine, 575, 340, 20, 420, (50, 300))
    too_soon = _update(machine, 580, 345, 40, 460, (50, 125))
    confirmed = _update(machine, 585, 350, 41, 470, (50, 125))

    assert crossing.state == BallShotState.CROSSED_OUTSIDE
    assert first.state == BallShotState.CROSSED_OUTSIDE
    assert too_soon.state == BallShotState.CROSSED_OUTSIDE
    assert confirmed.state == BallShotState.MISSED


def test_inside_center_crossing_can_continue_diagonally_and_become_made() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 545, 245, 2, 200, (100, -180))
    _update(machine, 545, 280, 3, 300, (0, 180))
    contact = _update(machine, 545, 310, 4, 400, (0, 350))
    made = _update(machine, 512, 338, 5, 500, (-180, 230))

    assert contact.state == BallShotState.CROSSED_INSIDE
    assert contact.crossed_rim_this_frame
    assert made.state == BallShotState.MADE
    assert made.outcome is not None
    assert made.outcome.result == "made"
    assert "center crossed inside opening" in made.outcome.evidence[-1]


def test_inside_center_crossing_from_other_side_becomes_made() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 455, 245, 2, 200, (-100, -180))
    _update(machine, 455, 280, 3, 300, (0, 180))
    contact = _update(machine, 455, 310, 4, 400, (0, 350))
    made = _update(machine, 488, 338, 5, 500, (180, 230))

    assert contact.state == BallShotState.CROSSED_INSIDE
    assert made.state == BallShotState.MADE
    assert made.outcome is not None
    assert made.outcome.result == "made"


def test_inside_center_crossing_that_bounces_away_becomes_missed() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 545, 245, 2, 200, (100, -180))
    _update(machine, 545, 280, 3, 300, (0, 180))
    contact = _update(machine, 545, 310, 4, 400, (0, 350))
    first_bounce = _update(machine, 548, 285, 5, 500, (180, -300))
    missed = _update(machine, 568, 260, 6, 600, (200, -250))

    assert contact.state == BallShotState.CROSSED_INSIDE
    assert first_bounce.state == BallShotState.CROSSED_INSIDE
    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None
    assert missed.outcome.result == "missed"


def test_predicted_points_cannot_confirm_a_make() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500, 250, 2, 200, (100, -150))
    _update(machine, 500, 280, 3, 300, (0, 200))
    crossed = _update(machine, 500, 310, 4, 400, (0, 350))
    assert crossed.state == BallShotState.CROSSED_INSIDE

    _update(machine, 500, 340, 5, 600, (0, 150), observed=False)
    unknown = _update(machine, 500, 360, 6, 1300, (0, 30), observed=False)

    assert unknown.state == BallShotState.UNKNOWN
    assert unknown.outcome is not None
    assert unknown.outcome.result == "unknown"


def test_ball_flight_continues_when_pose_is_missing() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500, 240, 2, 200, (100, -200), pose_phase=None)
    _update(machine, 500, 280, 3, 300, (0, 200), pose_phase=None)
    _update(machine, 500, 315, 4, 400, (0, 350), pose_phase=None)
    made = _update(machine, 500, 335, 5, 500, (0, 200), pose_phase=None)

    assert made.state == BallShotState.MADE


def test_nanotrack_evidence_is_labelled_separately_from_yolo() -> None:
    machine = BallShotStateMachine()
    detection, snapshot = _ball(100, 500, 0, 0)
    assert detection is not None
    detection.measurement_source = "nanotrack"
    snapshot.measurement_source = "nanotrack"

    result = machine.update(
        ball_detection=detection,
        ball_snapshot=snapshot,
        rim_detection=_rim(),
        wrist_xy=(100, 500),
        ankle_y=_STANDING_ANKLE_Y,
        pose_phase="rise",
        timestamp_ms=0,
    )

    assert result.tracking_status == BallTrackingStatus.TRACKED
    assert result.measurement_source == "nanotrack"


def test_moving_camera_uses_ball_position_relative_to_rim() -> None:
    machine = BallShotStateMachine()
    machine.dynamic_rim = True

    def moving_update(
        base_ball_xy,
        camera_offset,
        frame,
        timestamp_ms,
        velocity,
        pose_phase=None,
        wrist_base_xy=None,
    ):
        offset_x, offset_y = camera_offset
        ball_x = base_ball_xy[0] + offset_x
        ball_y = base_ball_xy[1] + offset_y
        detection, snapshot = _ball(
            ball_x,
            ball_y,
            frame,
            timestamp_ms,
            velocity,
        )
        wrist_xy = None
        if wrist_base_xy is not None:
            wrist_xy = (
                wrist_base_xy[0] + offset_x,
                wrist_base_xy[1] + offset_y,
            )
        rim = RimDetection(
            center_xy=(500.0 + offset_x, 300.0 + offset_y),
            bbox_xyxy=(
                450.0 + offset_x,
                300.0 + offset_y,
                550.0 + offset_x,
                430.0 + offset_y,
            ),
            confidence=0.9,
            frame_index=frame,
            timestamp_ms=timestamp_ms,
        )
        return machine.update(
            ball_detection=detection,
            ball_snapshot=snapshot,
            rim_detection=rim,
            wrist_xy=wrist_xy,
            # The feet are part of the scene, so they pan with the camera
            # exactly as the ball and the rim do.
            ankle_y=_STANDING_ANKLE_Y + offset_y,
            pose_phase=pose_phase,
            timestamp_ms=timestamp_ms,
        )

    moving_update(
        (100, 500), (0, 0), 0, 0, (0, 0),
        pose_phase="rise", wrist_base_xy=(100, 500),
    )
    released = moving_update(
        (200, 400), (20, 10), 1, 100, (700, -400),
        pose_phase="release", wrist_base_xy=(100, 500),
    )
    assert released.state == BallShotState.ASCENDING

    moving_update((500, 240), (40, 20), 2, 200, (300, -100))
    moving_update((500, 280), (60, 30), 3, 300, (200, 300))
    crossing = moving_update((500, 315), (80, 40), 4, 400, (200, 450))
    made = moving_update((502, 335), (100, 50), 5, 500, (220, 300))

    assert crossing.crossed_rim_this_frame
    assert made.state == BallShotState.MADE
    assert made.outcome is not None
    assert made.outcome.result == "made"


def _body_snapshot(
    timestamp_ms: int,
    phase: str,
    wrist_y: float = 1.0,
    wrist_height_ratio=None,
) -> FrameSnapshot:
    from phase_detection.features import KinematicFeatures

    return FrameSnapshot(
        timestamp_ms=timestamp_ms,
        angles={},
        shooting_side="right",
        phase=phase,
        features=KinematicFeatures(
            wrist_y=wrist_y,
            shoulder_y=1.40,
            hip_y_avg=0.95,
            nose_y=1.60,
            ankle_y_avg=0.10,
            ankle_baseline_y=0.10,
            wrist_height_ratio=wrist_height_ratio,
        ),
    )


def test_shot_tracker_waits_for_ball_after_body_finishes() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)

    # A credible motion: the wrist must actually travel and the ball must be
    # released, otherwise the candidate is discarded as a posture change.
    assert tracker.update("rise", _body_snapshot(0, "rise", 1.00)) is None
    assert tracker.update("rise", _body_snapshot(60, "rise", 1.20)) is None
    assert tracker.update("release", _body_snapshot(120, "release", 1.48)) is None
    assert tracker.update("recovery", _body_snapshot(180, "recovery", 1.05)) is None
    assert tracker.update("ready", _body_snapshot(240, "ready", 1.00)) is None
    assert tracker.shot_in_progress
    assert not tracker.capture_in_progress

    outcome = ShotOutcome(
        result="made",
        confidence=0.95,
        release_timestamp_ms=150,
        outcome_timestamp_ms=300,
        evidence=["synthetic inside crossing"],
        timeseries_summary={"trajectory_comparison": {}},
    )
    summary = tracker.update_ball_outcome(outcome, 300)

    assert summary is not None
    assert summary.outcome is outcome
    assert summary.pose_release_timestamp_ms == 120
    assert summary.ball_release_timestamp_ms == 150
    assert summary.release_disagreement_ms == 30
    assert summary.ball_minus_pose_release_ms == 30
    assert not tracker.shot_in_progress

    payload = shot_summary_to_dict(summary)
    assert payload["outcome"]["result"] == "made"
    assert payload["outcome"]["is_basket"] is True
    expected_timing = {
        "pose_release_timestamp_ms": 120,
        "ball_release_timestamp_ms": 150,
        "release_disagreement_ms": 30,
        "ball_minus_pose_release_ms": 30,
        "release_disagreement_limit_ms": 100,
        "release_timing_status": "aligned",
        "release_alignment_confidence": "normal",
    }
    assert payload["release_timing"] == expected_timing
    assert payload["outcome"]["release_timing"] == expected_timing
    assert payload["outcome"]["trajectory_comparison"] == expected_timing


def test_ball_outcome_can_finish_after_body_grace_when_pose_never_lands() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)
    tracker.update("rise", _body_snapshot(0, "rise", 1.00))
    tracker.update("rise", _body_snapshot(30, "rise", 1.20))
    tracker.update("release", _body_snapshot(60, "release", 1.48))

    outcome = ShotOutcome(
        result="missed",
        confidence=0.8,
        outcome_timestamp_ms=100,
        evidence=["synthetic outside crossing"],
    )
    assert tracker.update_ball_outcome(outcome, 100) is None
    summary = tracker.update_ball_outcome(outcome, 600)

    assert summary is not None
    assert summary.ended_early
    assert summary.outcome is outcome
    assert summary.pose_release_timestamp_ms == 60
    assert summary.ball_release_timestamp_ms is None
    assert summary.release_disagreement_ms is None
    assert summary.ball_minus_pose_release_ms is None

    payload = shot_summary_to_dict(summary)
    assert payload["release_timing"] == {
        "pose_release_timestamp_ms": 60,
        "ball_release_timestamp_ms": None,
        "release_disagreement_ms": None,
        "ball_minus_pose_release_ms": None,
        "release_disagreement_limit_ms": 100,
        "release_timing_status": "unavailable",
        "release_alignment_confidence": "unassessed",
    }


def test_large_release_disagreement_marks_trajectory_low_confidence() -> None:
    timing = release_timing_fields(
        pose_release_timestamp_ms=2333,
        ball_release_timestamp_ms=1833,
        disagreement_limit_ms=100,
    )

    assert timing["release_disagreement_ms"] == 500
    assert timing["ball_minus_pose_release_ms"] == -500
    assert timing["release_timing_status"] == "disagreed"
    assert timing["release_alignment_confidence"] == "low"


def test_final_pose_release_uses_full_shot_wrist_peak() -> None:
    tracker = ShotTracker()

    assert tracker.update(
        "rise", _body_snapshot(0, "rise", 1.00, 0.10)
    ) is None
    assert tracker.update(
        "rise", _body_snapshot(60, "rise", 1.20, 0.40)
    ) is None
    assert tracker.update(
        "release", _body_snapshot(120, "release", 1.48, 0.70)
    ) is None
    assert tracker.update(
        "recovery", _body_snapshot(180, "recovery", 1.05, 0.90)
    ) is None
    summary = tracker.update(
        "ready", _body_snapshot(240, "ready", 1.00, 0.30)
    )

    assert summary is not None
    # The live pose transition fired at 120 ms, but final coaching phase cuts
    # deliberately use the highest wrist over the complete captured attempt.
    assert summary.pose_release_timestamp_ms == 180
    assert summary.ball_release_timestamp_ms is None


def test_ball_release_recovers_pose_attempt_when_pose_fsm_misses_anchor() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)
    history = FrameBuffer(max_frames=30)
    tracker.attach_history(history)

    def update_pose(timestamp_ms, wrist_y, wrist_ratio, *, ball_release=None, outcome=None):
        snapshot = _body_snapshot(
            timestamp_ms,
            "ready",
            wrist_y,
            wrist_ratio,
        )
        history.push(snapshot)
        return tracker.update(
            "ready",
            snapshot,
            ball_outcome=outcome,
            ball_release_timestamp_ms=ball_release,
        )

    assert update_pose(0, 1.00, 0.10) is None
    assert update_pose(60, 1.20, 0.40) is None
    assert update_pose(120, 1.48, 0.75, ball_release=120) is None
    assert tracker.shot_in_progress
    assert update_pose(180, 1.55, 0.90) is None
    assert update_pose(240, 1.40, 0.70) is None
    assert update_pose(400, 1.10, 0.20) is None

    outcome = ShotOutcome(
        result="made",
        confidence=0.9,
        release_timestamp_ms=120,
        outcome_timestamp_ms=500,
        evidence=["synthetic ball-driven attempt"],
    )
    summary = update_pose(500, 1.00, 0.10, outcome=outcome)

    assert summary is not None
    assert summary.outcome is outcome
    assert summary.pose_release_timestamp_ms == 180
    assert summary.ball_release_timestamp_ms == 120
    assert summary.release_disagreement_ms == 60
    assert summary.release_alignment_confidence == "normal"
    assert tracker.shot_count == 1
    assert not tracker.shot_in_progress


def test_release_timeout_waits_for_required_ball_outcome() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)

    assert tracker.update("rise", _body_snapshot(0, "rise", 1.00)) is None
    assert tracker.update("release", _body_snapshot(100, "release", 1.48)) is None
    # The follow-through clock has expired, but a required ball outcome has
    # not arrived. This must mark the body complete without emitting a
    # pose-only duplicate summary.
    assert tracker.update(
        "recovery", _body_snapshot(1400, "recovery", 1.10)
    ) is None
    assert tracker.shot_in_progress
    assert not tracker.capture_in_progress

    outcome = ShotOutcome(
        result="missed",
        confidence=0.8,
        release_timestamp_ms=133,
        outcome_timestamp_ms=1500,
        evidence=["synthetic late outcome"],
    )
    summary = tracker.update_ball_outcome(outcome, 1500)
    assert summary is not None
    assert summary.outcome is outcome
    assert tracker.shot_count == 1


if __name__ == "__main__":
    test_clean_inside_crossing_becomes_made()
    test_outside_crossing_becomes_missed()
    test_miss_confirmation_uses_elapsed_time_not_frame_count()
    test_inside_center_crossing_can_continue_diagonally_and_become_made()
    test_inside_center_crossing_from_other_side_becomes_made()
    test_inside_center_crossing_that_bounces_away_becomes_missed()
    test_predicted_points_cannot_confirm_a_make()
    test_ball_flight_continues_when_pose_is_missing()
    test_nanotrack_evidence_is_labelled_separately_from_yolo()
    test_moving_camera_uses_ball_position_relative_to_rim()
    test_shot_tracker_waits_for_ball_after_body_finishes()
    test_ball_outcome_can_finish_after_body_grace_when_pose_never_lands()
    test_large_release_disagreement_marks_trajectory_low_confidence()
    test_final_pose_release_uses_full_shot_wrist_peak()
    test_ball_release_recovers_pose_attempt_when_pose_fsm_misses_anchor()
    test_release_timeout_waits_for_required_ball_outcome()
    print("All ball shot state-machine tests passed.")
