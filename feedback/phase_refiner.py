"""Assign coaching phases to a captured shot, after the fact.

WHY THIS EXISTS
---------------
The live state machine (phase_detection.detector) tracks four states, because
finding a shot reliably means having as few places to get stuck as possible.
The coaching report needs more than four headings: "your knee bend" and "your
follow-through" are different advice.

This module bridges the two. It receives one shot's frames once the whole
motion is in hand, and works out where each coaching phase begins and ends.

THE POINT IS THAT IT LOOKS BACKWARDS
------------------------------------
Live detection has to decide "is the knee bending NOW?" from a velocity, a
threshold and a hysteresis window, and if that decision goes wrong the state
machine sits in the wrong state for the rest of the shot. Looking backwards,
the same question is "which frame had the smallest knee angle?" -- an argmin
over an array. It cannot fail to produce an answer, it cannot get stuck, and
it needs no threshold at all.

That is why the phases the player is shown are computed here and not there.

BOUNDARIES
----------
Six segments, from five cut points, each found from the signal that actually
defines it:

    loading         everything up to the bottom of the dip, including the
                    stance before the knee starts to bend
    ball_lift       upward drive, feet still on the floor
    jump            feet off the floor          (omitted if the player
                                                 never left the floor)
    release         the ball leaves the hand
    follow_through  the finish is held
    landing         back on the floor / hand back down

Cut points are forced into non-decreasing order, so a signal that reads out of
sequence collapses its segment to nothing rather than scrambling the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from phase_detection.phases import CORE_ANCHOR
from utils.config_loader import load_yaml

_MODEL = load_yaml("phase_model.yaml")
_CFG = _MODEL.get("refinement", {}) or {}

# How much whole-body elevation counts as leaving the floor at all, as a
# fraction of the player's own on-screen height. Below this there is no `jump`
# phase and no `landing` driven by the feet -- which is correct for a set shot.
_MIN_TAKEOFF_RATIO = float(_CFG.get("min_takeoff_ratio", 0.05))

# Where within the rise the feet are taken to have left the floor, as a
# fraction of that shot's own peak elevation. Relative rather than absolute so
# a small hop and a full jump are both cut at the same point in their arc.
_TAKEOFF_FRACTION = float(_CFG.get("takeoff_fraction", 0.35))

# The finish is held briefly before the arm drops, so the descent is not
# looked for immediately after release.
_MIN_FOLLOW_THROUGH_S = float(_CFG.get("min_follow_through_s", 0.15))


@dataclass
class PhaseCuts:
    """Frame indices where each coaching phase begins. Half-open segments."""

    load_start: int
    lift_start: int
    takeoff: int
    release_start: int
    release_end: int
    landing_start: int
    total: int
    jumped: bool

    def as_labels(self) -> List[str]:
        # Everything before the lift is `loading`. There is no separate resting
        # phase any more: it carried no rule, and on real clips it was often a
        # single frame or none at all. See phase_model.yaml.
        labels = ["loading"] * self.total
        _fill(labels, self.load_start, self.lift_start, "loading")
        _fill(labels, self.lift_start, self.takeoff, "ball_lift")
        _fill(labels, self.takeoff, self.release_start, "jump")
        _fill(labels, self.release_start, self.release_end, "release")
        _fill(labels, self.release_end, self.landing_start, "follow_through")
        _fill(labels, self.landing_start, self.total, "landing")
        return labels


def _fill(labels: List[str], start: int, end: int, value: str) -> None:
    for i in range(max(0, start), min(len(labels), end)):
        labels[i] = value


# ------------------------------------------------------------------ signals
def _wrist_level(f) -> Optional[float]:
    """How high the shooting hand is, in whichever space is trustworthy.

    World landmarks read a fabricated 0.0 when MediaPipe's visibility gate
    rejects the wrist, which filming from behind triggers on every frame. The
    image-space ratio survives that gate, so it is preferred.
    """
    if f is None:
        return None
    if f.wrist_height_ratio is not None:
        return f.wrist_height_ratio
    if f.wrist_world_valid:
        return f.wrist_y
    return None


def _argmin(values: Sequence[Optional[float]], lo: int, hi: int) -> Optional[int]:
    best_i, best_v = None, None
    for i in range(max(0, lo), min(len(values), hi)):
        v = values[i]
        if v is None:
            continue
        if best_v is None or v < best_v:
            best_i, best_v = i, v
    return best_i


def _argmax(values: Sequence[Optional[float]], lo: int, hi: int) -> Optional[int]:
    best_i, best_v = None, None
    for i in range(max(0, lo), min(len(values), hi)):
        v = values[i]
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_i, best_v = i, v
    return best_i


def _release_span(
    frames: Sequence,
    wrist: Sequence[Optional[float]],
    event_index: Optional[int] = None,
) -> tuple:
    """The frames during which the ball left the hand.

    `event_index` wins when the caller has one. Offline segmentation locates
    the shooting event as the peak of the hand's own trajectory across the
    whole video, which is a better answer than anything derivable from inside
    the window -- and, critically, it does not come from the live detector.

    That matters because the frames still carry the live detector's labels. On
    an attempt where it entered `release` early, every frame from the window's
    start was marked, `release_start` landed at 0, and the dip and the lift had
    nowhere to live: a 2.4 s shot reported three phases out of seven, with no
    knee bend for the loading rules to score.

    The state machine remains the source when no event is supplied, which is
    the live path, where its anchor is the only evidence available.
    """
    if event_index is not None:
        clamped = max(0, min(int(event_index), len(frames) - 1))
        return clamped, min(len(frames), clamped + 1)

    marked = [i for i, s in enumerate(frames) if s.phase == CORE_ANCHOR]
    if marked:
        return marked[0], marked[-1] + 1

    peak = _argmax(wrist, 0, len(frames))
    if peak is None:
        n = len(frames)
        return n, n
    return peak, min(len(frames), peak + 1)


def compute_cuts(
    frames: Sequence, event_index: Optional[int] = None
) -> PhaseCuts:
    """Locate every coaching-phase boundary in one captured shot."""
    n = len(frames)
    if n == 0:
        return PhaseCuts(0, 0, 0, 0, 0, 0, 0, False)

    feats = [s.features for s in frames]
    wrist = [_wrist_level(f) for f in feats]
    knee = [(f.knee_angle if f is not None else None) for f in feats]
    rise = [(f.body_rise_ratio if f is not None else None) for f in feats]

    release_start, release_end = _release_span(frames, wrist, event_index)

    # --- the dip -----------------------------------------------------------
    # Bottom of the countermovement, by definition the smallest knee angle
    # before the ball goes. The lift begins there.
    dip_bottom = _argmin(knee, 0, release_start)
    if dip_bottom is not None:
        # Loading ends AT the bottom of the dip and the lift begins the frame
        # after, so the deepest bend falls inside `loading` -- which is where
        # the knee and hip rules run. Off by one frame here and the single
        # measurement those rules exist to score sits under the next heading.
        lift_start = dip_bottom + 1
        # The dip begins where the knee was straightest on the way in.
        load_start = _argmax(knee, 0, dip_bottom + 1)
        if load_start is None:
            load_start = 0
    else:
        # No knee angle anywhere -- fall back to the hand. Its lowest point
        # before release is where the upward drive began. There is nothing
        # left to locate a dip with, so no loading phase is claimed.
        fallback = _argmin(wrist, 0, release_start)
        lift_start = fallback if fallback is not None else 0
        load_start = lift_start

    # --- leaving the floor -------------------------------------------------
    # Searched across the WHOLE captured shot, not up to the release. A player
    # is highest at or just after the ball goes and comes down through the
    # follow-through, so a window that stops at the release samples the way up
    # and misses the top -- and then reports a jump shot with no `jump` phase
    # and no `landing`, which is the one description that is certainly wrong.
    #
    # Widening it is safe HERE in a way it was not for the classifier, which
    # was tried and reverted: this decides only which frames get which heading,
    # never which type of shot it was. A slightly generous airborne window
    # mislabels a frame; a generous classifier mislabels the shot.
    peak_rise = 0.0
    for i in range(max(0, lift_start), n):
        v = rise[i]
        if v is not None and v > peak_rise:
            peak_rise = v

    jumped = peak_rise >= _MIN_TAKEOFF_RATIO
    takeoff = release_start
    if jumped:
        bar = _TAKEOFF_FRACTION * peak_rise
        for i in range(max(0, lift_start), min(n, release_start)):
            v = rise[i]
            if v is not None and v >= bar:
                takeoff = i
                break

    # --- the end of the finish ---------------------------------------------
    landing_start = _find_landing(frames, feats, rise, release_end, jumped, peak_rise)

    # Force the timeline to run forwards. A signal that reads out of order --
    # a knee minimum recorded after the release because the legs tucked in
    # flight, say -- collapses its own segment to nothing instead of
    # scrambling every segment after it.
    cuts = [load_start, lift_start, takeoff, release_start, release_end, landing_start]
    running = 0
    ordered = []
    for c in cuts:
        running = max(running, min(int(c), n))
        ordered.append(running)

    return PhaseCuts(*ordered, total=n, jumped=jumped)


def _find_landing(
    frames: Sequence,
    feats: Sequence,
    rise: Sequence[Optional[float]],
    release_end: int,
    jumped: bool,
    peak_rise: float,
) -> int:
    """Where the finish ends and the landing begins.

    Two different questions, because the two shot types end differently. A
    jump shot ends when the feet are back on the floor. A set shot never left
    the floor, so its finish ends when the shooting hand comes back down.
    """
    n = len(frames)
    if release_end >= n:
        return n

    start_ms = frames[min(release_end, n - 1)].timestamp_ms

    for i in range(release_end, n):
        f = feats[i]
        if f is None:
            continue
        elapsed_s = (frames[i].timestamp_ms - start_ms) / 1000.0
        if elapsed_s < _MIN_FOLLOW_THROUGH_S:
            continue
        if jumped:
            v = rise[i]
            if v is not None and v < _TAKEOFF_FRACTION * peak_rise:
                return i
        elif _hand_has_dropped(f):
            return i
    return n


def _hand_has_dropped(f) -> bool:
    """Has the shooting hand come back down below the shoulder?"""
    if f.wrist_above_shoulder_ratio is not None:
        return f.wrist_above_shoulder_ratio < 0.0
    if f.wrist_world_valid:
        return f.wrist_y < f.shoulder_y
    return False


def hold_duration_s(
    frames: Sequence,
    event_index: Optional[int] = None,
) -> Optional[float]:
    """Seconds the shooting arm stayed up after the ball left.

    WHY THIS IS NOT THE LENGTH OF THE `follow_through` PHASE
    -------------------------------------------------------
    Measured on salah_video, that phase was exactly 5 frames -- 0.17 s -- on
    every single shot. Not approximately: identically, five out of five. It is
    pinned to `min_follow_through_s` (0.15 s) at one end and to the LANDING at
    the other, and for a shot with any elevation the feet start coming down
    immediately, so the phase closes at the first frame it is allowed to.

    That makes the phase a poor witness for the thing it is named after. A
    player who holds the finish for two seconds and one who drops the arm at
    once produce the same 0.17 s window, and `follow_through_elbow` therefore
    measures the elbow shortly after release rather than whether anything was
    held.

    The arm and the feet are independent. This measures the arm directly: from
    the shooting event until the shooting hand falls back below the shoulder,
    searched across every frame captured after the release regardless of which
    phase they were labelled.

    Returns None when the hand never came down inside the captured window --
    "still up when the clip ended" is not a duration, and reporting the clip
    length instead would turn a cut-off recording into a good follow-through.
    """
    n = len(frames)
    if n == 0:
        return None

    wrist = [_wrist_level(s.features) for s in frames]
    start, _ = _release_span(frames, wrist, event_index)
    if start >= n:
        return None

    release_ms = frames[start].timestamp_ms
    for i in range(start + 1, n):
        f = frames[i].features
        if f is None:
            continue
        if _hand_has_dropped(f):
            return max(0.0, (frames[i].timestamp_ms - release_ms) / 1000.0)
    return None


def refine_phases(frames: Sequence, event_index: Optional[int] = None) -> List[str]:
    """Coaching phase for every frame of one captured shot."""
    return compute_cuts(frames, event_index).as_labels()
