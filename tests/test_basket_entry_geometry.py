"""Focused checks for clean, contact, and invalid rim entries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallDetection, BallSnapshot, RimDetection
from ball.shot_state_machine import BallShotState, BallShotStateMachine


def _rim(frame: int, timestamp_ms: int) -> RimDetection:
    return RimDetection(
        center_xy=(500.0, 300.0),
        bbox_xyxy=(450.0, 300.0, 550.0, 430.0),
        confidence=0.95,
        frame_index=frame,
        timestamp_ms=timestamp_ms,
    )


def _update(
    machine: BallShotStateMachine,
    x: float,
    y: float,
    frame: int,
    velocity=(0.0, 0.0),
    pose_phase=None,
    wrist_xy=None,
):
    timestamp_ms = frame * 100
    detection = BallDetection(
        center_xy=(x, y),
        bbox_xyxy=(x - 10.0, y - 10.0, x + 10.0, y + 10.0),
        confidence=0.95,
        frame_index=frame,
        timestamp_ms=timestamp_ms,
    )
    snapshot = BallSnapshot(
        timestamp_ms=timestamp_ms,
        frame_index=frame,
        center_xy=(x, y),
        confidence=0.95,
        velocity_xy=velocity,
    )
    return machine.update(
        ball_detection=detection,
        ball_snapshot=snapshot,
        rim_detection=_rim(frame, timestamp_ms),
        wrist_xy=wrist_xy,
        ankle_y=800.0,
        pose_phase=pose_phase,
        timestamp_ms=timestamp_ms,
        player_height_px=700.0,
    )


def _release(machine: BallShotStateMachine) -> None:
    _update(
        machine,
        100.0,
        500.0,
        0,
        pose_phase="rise",
        wrist_xy=(100.0, 500.0),
    )
    released = _update(
        machine,
        200.0,
        400.0,
        1,
        velocity=(500.0, -500.0),
        pose_phase="release",
        wrist_xy=(100.0, 500.0),
    )
    assert released.state == BallShotState.ASCENDING


def test_clean_entry_requires_downward_contour_fit_then_below_confirmation():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500.0, 270.0, 2, velocity=(0.0, -100.0))
    crossing = _update(machine, 500.0, 310.0, 3, velocity=(0.0, 300.0))
    made = _update(machine, 500.0, 330.0, 4, velocity=(0.0, 200.0))

    assert crossing.crossed_rim_this_frame
    assert crossing.state == BallShotState.CROSSED_INSIDE
    assert crossing.crossing_timestamp_ms is not None
    assert made.state == BallShotState.MADE
    assert made.outcome is not None and made.outcome.result == "made"
    assert made.outcome.entry_timestamp_ms == crossing.crossing_timestamp_ms


def test_clean_entry_can_continue_diagonally_without_a_net():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500.0, 270.0, 2, velocity=(100.0, -100.0))
    crossing = _update(machine, 530.0, 310.0, 3, velocity=(300.0, 300.0))
    made = _update(machine, 575.0, 335.0, 4, velocity=(450.0, 250.0))

    assert crossing.crossed_rim_this_frame
    assert crossing.state == BallShotState.CROSSED_INSIDE
    assert made.state == BallShotState.MADE
    assert made.outcome is not None and made.outcome.result == "made"
    assert "clean inside crossing" in made.outcome.evidence[-1]


def test_ball_coming_from_below_cannot_become_made():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 500.0, 330.0, 2, velocity=(0.0, -200.0))
    _update(machine, 500.0, 305.0, 3, velocity=(0.0, -150.0))
    result = _update(machine, 500.0, 335.0, 4, velocity=(0.0, 200.0))

    assert not result.crossed_rim_this_frame
    assert result.state != BallShotState.MADE


def test_rim_contact_can_continue_diagonally_after_inner_entry():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 545.0, 270.0, 2, velocity=(0.0, -100.0))
    contact = _update(machine, 545.0, 310.0, 3, velocity=(0.0, 300.0))
    # The crossing center was inside the physical opening. By the time the
    # ball is sufficiently below, a netless rim allows it to continue outside
    # both the inner radius and the old vertical net channel.
    made = _update(machine, 580.0, 330.0, 4, velocity=(350.0, 200.0))

    assert contact.crossed_rim_this_frame
    assert contact.state == BallShotState.RIM_CONTACT
    assert made.state == BallShotState.MADE
    assert made.outcome is not None and made.outcome.result == "made"
    assert "after rim contact" in made.outcome.evidence[-1]


def test_verified_contact_crossing_that_rebounds_up_is_missed():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 545.0, 270.0, 2, velocity=(0.0, -100.0))
    contact = _update(machine, 545.0, 310.0, 3, velocity=(0.0, 300.0))
    _update(machine, 560.0, 285.0, 4, velocity=(150.0, -300.0))
    missed = _update(machine, 575.0, 270.0, 5, velocity=(150.0, -200.0))

    assert contact.crossed_rim_this_frame
    assert contact.state == BallShotState.RIM_CONTACT
    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None and missed.outcome.result == "missed"


def test_rim_contact_requires_the_actual_inner_opening():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 550.0, 270.0, 2, velocity=(0.0, -100.0))
    contact = _update(machine, 550.0, 310.0, 3, velocity=(0.0, 300.0))
    # Offset 50 is inside the old 1.2x net channel (57 px), but outside
    # the physical inner opening (47.5 px).
    _update(machine, 550.0, 335.0, 4, velocity=(50.0, 250.0))
    missed = _update(machine, 550.0, 350.0, 5, velocity=(0.0, 150.0))

    assert contact.crossed_rim_this_frame
    assert contact.state == BallShotState.RIM_CONTACT
    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None and missed.outcome.result == "missed"


def test_projected_edge_overlap_does_not_turn_airball_into_make():
    machine = BallShotStateMachine()
    _release(machine)

    # The ball circle visually overlaps the right rim edge, but its center
    # crosses the rim plane beyond the allowed contact channel.
    _update(machine, 560.0, 285.0, 2, velocity=(0.0, -100.0))
    crossing = _update(machine, 560.0, 305.0, 3, velocity=(0.0, 300.0))
    # A front/behind airball can then project underneath the opening. Once its
    # rim-plane crossing was outside, these later points cannot upgrade it.
    under_rim = _update(machine, 540.0, 330.0, 4, velocity=(-200.0, 250.0))
    missed = _update(machine, 525.0, 345.0, 5, velocity=(-150.0, 150.0))

    assert crossing.crossed_rim_this_frame
    assert crossing.state == BallShotState.CROSSED_OUTSIDE
    assert under_rim.state == BallShotState.CROSSED_OUTSIDE
    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None and missed.outcome.result == "missed"
    assert not machine.saw_rim_contact


def test_outside_downward_entry_becomes_missed():
    machine = BallShotStateMachine()
    _release(machine)

    _update(machine, 570.0, 270.0, 2, velocity=(0.0, -100.0))
    crossing = _update(machine, 570.0, 310.0, 3, velocity=(0.0, 300.0))
    _update(machine, 580.0, 330.0, 4, velocity=(100.0, 200.0))
    missed = _update(machine, 590.0, 345.0, 5, velocity=(100.0, 150.0))

    assert crossing.crossed_rim_this_frame
    assert crossing.state == BallShotState.CROSSED_OUTSIDE
    assert missed.state == BallShotState.MISSED
    assert missed.outcome is not None and missed.outcome.result == "missed"


if __name__ == "__main__":
    test_clean_entry_requires_downward_contour_fit_then_below_confirmation()
    test_clean_entry_can_continue_diagonally_without_a_net()
    test_ball_coming_from_below_cannot_become_made()
    test_rim_contact_can_continue_diagonally_after_inner_entry()
    test_verified_contact_crossing_that_rebounds_up_is_missed()
    test_rim_contact_requires_the_actual_inner_opening()
    test_projected_edge_overlap_does_not_turn_airball_into_make()
    test_outside_downward_entry_becomes_missed()
    print("Basket entry geometry tests passed.")
