"""Synthetic tests for release, rim crossing, and shot completion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallDetection, BallSnapshot, RimDetection, ShotOutcome
from ball.shot_state_machine import BallShotState, BallShotStateMachine
from feedback.console import shot_summary_to_dict
from feedback.shot_tracker import ShotTracker
from utils.frame_buffer import FrameSnapshot


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
        pose_phase=pose_phase,
        timestamp_ms=timestamp_ms,
    )


def _release(machine: BallShotStateMachine) -> None:
    _update(
        machine,
        100,
        500,
        0,
        0,
        (0, 0),
        pose_phase="ball_lift",
        wrist_xy=(100, 500),
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


def test_predicted_points_cannot_confirm_a_make() -> None:
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500, 250, 2, 200, (100, -150))
    _update(machine, 500, 280, 3, 300, (0, 200))
    crossed = _update(machine, 500, 315, 4, 400, (0, 350))
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


def _body_snapshot(timestamp_ms: int, phase: str) -> FrameSnapshot:
    return FrameSnapshot(
        timestamp_ms=timestamp_ms,
        angles={},
        shooting_side="right",
        phase=phase,
    )


def test_shot_tracker_waits_for_ball_after_body_finishes() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)

    assert tracker.update("loading", _body_snapshot(0, "loading")) is None
    assert tracker.update("landing", _body_snapshot(100, "landing")) is None
    assert tracker.update("ready_stance", _body_snapshot(200, "ready_stance")) is None
    assert tracker.shot_in_progress
    assert not tracker.capture_in_progress

    outcome = ShotOutcome(
        result="made",
        confidence=0.95,
        outcome_timestamp_ms=300,
        evidence=["synthetic inside crossing"],
    )
    summary = tracker.update_ball_outcome(outcome, 300)

    assert summary is not None
    assert summary.outcome is outcome
    assert not tracker.shot_in_progress

    payload = shot_summary_to_dict(summary)
    assert payload["outcome"]["result"] == "made"
    assert payload["outcome"]["is_basket"] is True


def test_ball_outcome_can_finish_after_body_grace_when_pose_never_lands() -> None:
    tracker = ShotTracker()
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)
    tracker.update("loading", _body_snapshot(0, "loading"))

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


if __name__ == "__main__":
    test_clean_inside_crossing_becomes_made()
    test_outside_crossing_becomes_missed()
    test_predicted_points_cannot_confirm_a_make()
    test_ball_flight_continues_when_pose_is_missing()
    test_shot_tracker_waits_for_ball_after_body_finishes()
    test_ball_outcome_can_finish_after_body_grace_when_pose_never_lands()
    print("All ball shot state-machine tests passed.")
