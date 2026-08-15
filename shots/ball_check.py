"""Ask the ball whether a candidate was a shot -- at the peak, and nowhere else.

WHY THE BALL VERIFIES INSTEAD OF DETECTS

Tracking a basketball through real practice footage is hard: the object is
small, motion-blurred at exactly the moment that matters, frequently occluded
by the shooter's own hands, and often a similar colour to the court. Building
detection on it means every tracking failure costs a whole shot.

Turned around, the same unreliable signal becomes safe. Body-side peak
detection proposes candidates; the ball is asked one question at each one, and
a tracking failure produces UNDECIDED rather than a missing shot. The system
degrades into "we could not tell", which is a state a coaching report can
honestly show.

It is also a clean division of labour: the pose side hands over a short list of
timestamps to look at, not a whole video to follow.

THE QUESTION

Around the peak, did the ball separate from the shooting hand and keep going?

  a shot   the ball leaves the hand and the gap grows, upward and away
  a catch  the ball arrives at the hand and the gap closes

A catch is the same body movement as a shot -- hands up, arms extended, hands
down -- which is why the body alone cannot separate them, and why `video_01`
reported four attempts where a person sees two. The ball's direction of travel
is what differs, and it differs unmistakably.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class BallVerdict(str, Enum):
    SHOT = "shot"
    CATCH = "catch"
    UNDECIDED = "undecided"


# How far either side of the peak to look. Long enough to see the ball travel,
# short enough that the next rep cannot leak in.
WINDOW_MS = 700

# Fewer observed points than this and there is nothing to judge. Interpolated
# points do not count: they are the tracker's guess, and a guess cannot confirm
# a shot. This mirrors `test_predicted_points_cannot_confirm_a_make` on the
# ball side, which draws the same line for the same reason.
MIN_OBSERVED_POINTS = 3

# How much the ball-to-wrist gap must grow, in body heights, before separation
# is called. A hand opening around a held ball moves it a few centimetres.
MIN_SEPARATION_RATIO = 0.15

# The ball must also be going UP at release, in body heights. Without this a
# ball dropped or handed downward would read as separation.
MIN_RISE_RATIO = 0.05


@dataclass(frozen=True)
class BallEvidence:
    """What the ball did around one candidate peak."""

    verdict: BallVerdict
    separation_ratio: Optional[float]
    rise_ratio: Optional[float]
    observed_points: int
    note: str


def _nearest(points: Sequence[Tuple[int, float, float]], target_ms: int):
    best = None
    for ts, x, y in points:
        gap = abs(ts - target_ms)
        if best is None or gap < best[0]:
            best = (gap, ts, x, y)
    return None if best is None else best[1:]


def verify_peak(
    ball_points: Sequence[Tuple[int, float, float]],
    wrist_points: Sequence[Tuple[int, float, float]],
    body_height_px: Optional[float],
    peak_ms: int,
    window_ms: int = WINDOW_MS,
) -> BallEvidence:
    """Decide whether the ball left the hand around `peak_ms`.

    `ball_points` and `wrist_points` are `(timestamp_ms, x, y)` in pixels, and
    only OBSERVED ball positions should be passed -- interpolated ones are the
    tracker filling a gap, and cannot serve as evidence.

    Distances are divided by the player's on-screen height so the thresholds
    hold at any camera distance, matching how every other measurement in this
    package is expressed.
    """
    if not body_height_px or body_height_px <= 0:
        return BallEvidence(
            BallVerdict.UNDECIDED, None, None, 0, "no body scale to measure against"
        )

    after = [p for p in ball_points if peak_ms <= p[0] <= peak_ms + window_ms]
    before = [p for p in ball_points if peak_ms - window_ms <= p[0] < peak_ms]
    if len(after) < MIN_OBSERVED_POINTS:
        return BallEvidence(
            BallVerdict.UNDECIDED,
            None,
            None,
            len(after),
            f"ball seen on {len(after)} frames after the peak, need "
            f"{MIN_OBSERVED_POINTS}",
        )

    at_peak = _nearest(wrist_points, peak_ms)
    if at_peak is None:
        return BallEvidence(
            BallVerdict.UNDECIDED, None, None, len(after), "shooting wrist not seen"
        )
    _, wrist_x, wrist_y = at_peak

    def gap(point) -> float:
        _, x, y = point
        return ((x - wrist_x) ** 2 + (y - wrist_y) ** 2) ** 0.5 / body_height_px

    start_gap = gap(after[0])
    end_gap = max(gap(p) for p in after)
    separation = end_gap - start_gap

    # Image y grows downward, so rising is a DECREASE.
    rise = (after[0][2] - min(p[2] for p in after)) / body_height_px

    if separation >= MIN_SEPARATION_RATIO and rise >= MIN_RISE_RATIO:
        return BallEvidence(
            BallVerdict.SHOT,
            separation,
            rise,
            len(after),
            f"ball left the hand and travelled {separation:.2f} body-heights "
            f"away while rising {rise:.2f}",
        )

    # A ball closing on the hand while the arms come down is a catch. Requiring
    # the approach to be visible BEFORE the peak keeps a shot whose flight was
    # simply lost from being called one.
    if before:
        approach = gap(before[0]) - gap(before[-1])
        if approach >= MIN_SEPARATION_RATIO and separation < MIN_SEPARATION_RATIO:
            return BallEvidence(
                BallVerdict.CATCH,
                separation,
                rise,
                len(after),
                f"ball closed {approach:.2f} body-heights onto the hand and did "
                "not leave again",
            )

    return BallEvidence(
        BallVerdict.UNDECIDED,
        separation,
        rise,
        len(after),
        f"ball moved {separation:.2f} body-heights and rose {rise:.2f}; too "
        "little to call either way",
    )
