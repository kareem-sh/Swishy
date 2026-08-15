"""Unit tests for ball-holder association (no ML runtime required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallHolderStatus, PlayerPoseCandidate, ShooterSelectionState
from player.ball_holder import BallHolderTracker, _score_candidate, best_candidate_for_ball


def _candidate(
    player_id: int,
    *,
    left_wrist=None,
    right_wrist=None,
    body_center=None,
    feet_midpoint=(100.0, 200.0),
    confidence=1.0,
) -> PlayerPoseCandidate:
    return PlayerPoseCandidate(
        player_id=player_id,
        left_wrist_xy=left_wrist,
        right_wrist_xy=right_wrist,
        body_center_xy=body_center or (100.0, 100.0),
        feet_midpoint_xy=feet_midpoint,
        confidence=confidence,
    )


def test_wrist_proximity_wins_over_body_center():
    ball_xy = (120.0, 110.0)
    near_wrist = _candidate(0, left_wrist=(118.0, 108.0), body_center=(300.0, 300.0))
    far_body = _candidate(1, left_wrist=(250.0, 250.0), body_center=(122.0, 112.0))
    best = best_candidate_for_ball([near_wrist, far_body], ball_xy, ball_confidence=0.9)
    assert best is not None
    assert best.player_id == 0


def test_scoring_prefers_visible_wrist():
    ball_xy = (50.0, 50.0)
    left_only = _candidate(0, left_wrist=(52.0, 51.0))
    no_wrist = _candidate(1, body_center=(51.0, 49.0), confidence=1.0)
    assert _score_candidate(left_only, ball_xy, 0.8) > _score_candidate(no_wrist, ball_xy, 0.8)


def test_shooter_lock_requires_confirmation():
    tracker = BallHolderTracker(confirm_frames=3, confirm_threshold=0.70)
    ball_xy = (100.0, 100.0)
    candidate = _candidate(0, left_wrist=(101.0, 99.0))

    first = tracker.update(ball_xy, [candidate], ball_confidence=0.9)
    assert first is not None
    assert first.tracking_status == BallHolderStatus.TENTATIVE.value

    second = tracker.update(ball_xy, [candidate], ball_confidence=0.9)
    third = tracker.update(ball_xy, [candidate], ball_confidence=0.9)
    assert third is not None
    assert third.tracking_status == BallHolderStatus.CONFIDENT.value
    assert third.shooter_state == ShooterSelectionState.CONFIRMED_SHOOTER.value


def test_release_uses_locked_shooter_not_new_candidate():
    tracker = BallHolderTracker(confirm_frames=1, confirm_threshold=0.50)
    ball_xy = (100.0, 100.0)
    shooter = _candidate(0, left_wrist=(101.0, 99.0), feet_midpoint_xy=(100.0, 220.0))
    other = _candidate(1, left_wrist=(400.0, 400.0), feet_midpoint_xy=(400.0, 500.0))

    tracker.update(ball_xy, [shooter], ball_confidence=0.9)
    released = tracker.update(ball_xy, [other, shooter], ball_confidence=0.9, released=True)
    assert released is not None
    assert released.player_id == 0
    assert released.shooter_state == ShooterSelectionState.RELEASED.value
