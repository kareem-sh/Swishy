"""The ball verifies a candidate; it never detects one.

Every case here checks the same property from a different angle: an
unreliable ball signal must degrade to UNDECIDED, never to a lost shot and
never to a confident wrong answer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shots.ball_check import (  # noqa: E402
    MIN_OBSERVED_POINTS,
    BallVerdict,
    verify_peak,
)

BODY = 300.0  # player 300 px tall on screen
WRIST = [(t, 500.0, 400.0) for t in range(-400, 500, 33)]


def _flight(peak_ms=0, step=40, n=8, rising=True):
    """Ball leaving the hand: away horizontally and up the frame."""
    return [
        (peak_ms + i * 33, 500.0 + i * step, 400.0 - (i * step if rising else 0))
        for i in range(n)
    ]


def test_a_ball_that_leaves_the_hand_and_rises_is_a_shot():
    result = verify_peak(_flight(), WRIST, BODY, 0)
    assert result.verdict is BallVerdict.SHOT
    assert result.separation_ratio > 0


def test_a_ball_arriving_at_the_hand_is_a_catch():
    incoming = [(-400 + i * 33, 500.0 + (10 - i) * 40, 400.0) for i in range(11)]
    settled = [(i * 33, 505.0, 400.0) for i in range(8)]
    result = verify_peak(incoming + settled, WRIST, BODY, 0)
    assert result.verdict is BallVerdict.CATCH


def test_too_few_observed_points_is_undecided_not_a_lost_shot():
    result = verify_peak(_flight(n=MIN_OBSERVED_POINTS - 1), WRIST, BODY, 0)
    assert result.verdict is BallVerdict.UNDECIDED
    assert "need" in result.note


def test_no_ball_at_all_is_undecided():
    assert verify_peak([], WRIST, BODY, 0).verdict is BallVerdict.UNDECIDED


def test_missing_body_scale_is_undecided():
    """Thresholds are in body heights, so without a body there is no measure."""
    assert verify_peak(_flight(), WRIST, None, 0).verdict is BallVerdict.UNDECIDED
    assert verify_peak(_flight(), WRIST, 0.0, 0).verdict is BallVerdict.UNDECIDED


def test_missing_wrist_is_undecided():
    assert verify_peak(_flight(), [], BODY, 0).verdict is BallVerdict.UNDECIDED


def test_a_ball_that_barely_moves_is_undecided_not_a_shot():
    held = [(i * 33, 500.0 + i, 400.0) for i in range(8)]
    result = verify_peak(held, WRIST, BODY, 0)
    assert result.verdict is BallVerdict.UNDECIDED


def test_a_ball_moving_away_downward_is_not_a_shot():
    """Handing the ball down separates it from the wrist too."""
    dropped = [(i * 33, 500.0 + i * 40, 400.0 + i * 40) for i in range(8)]
    assert verify_peak(dropped, WRIST, BODY, 0).verdict is not BallVerdict.SHOT


def test_the_window_excludes_a_neighbouring_attempt():
    """A flight belonging to the next rep must not confirm this one."""
    far_later = _flight(peak_ms=3000)
    assert verify_peak(far_later, WRIST, BODY, 0).verdict is BallVerdict.UNDECIDED


def test_scale_invariance_a_further_away_player_reads_the_same():
    near = verify_peak(_flight(step=40), WRIST, BODY, 0)
    half = [(t, 500.0 + (x - 500.0) / 2, 400.0 + (y - 400.0) / 2)
            for t, x, y in _flight(step=40)]
    far = verify_peak(half, WRIST, BODY / 2, 0)
    assert near.verdict is far.verdict is BallVerdict.SHOT
    assert abs(near.separation_ratio - far.separation_ratio) < 1e-6
