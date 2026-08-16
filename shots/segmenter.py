"""Find the shots in a captured video, once, with the whole video in hand.

This replaces a live state machine. The machine had to decide at every frame,
with no view of what came next, and so needed timeouts, hysteresis, a latch, a
refractory timer and a backwards refill -- every one of them there to
compensate for not seeing the future. None of that is needed here: the video is
complete before the first decision is taken.

THE SIGNAL

`wrist_height_ratio`: the shooting wrist's height above the hips, divided by
the player's own on-screen height. Image space, so it survives the visibility
gate that zeroes world-space wrists on rear-facing footage, and in body-height
units, so it is invariant to zoom and camera distance.

FINDING THE SHOTS -- PROMINENCE, NOT A THRESHOLD

A height threshold asks "is the hand above X?", which is true of a shot, a
catch, a scratch and a stretch alike. Prominence asks "how far did the hand
rise above its OWN surroundings?" -- the drop you would have to descend before
you could climb to any higher peak. It has no zero point and no baseline, so it
survives a drifting signal, and it is the rigorous form of the running floor
the old detector hand-rolled.

Measured on video8, whose ten shot windows are frame-exact in
single_shot/manifest.json: the nine visible shots cluster between 0.636 and
0.705 prominence, while the loudest non-shot reaches 0.445. Any threshold from
0.45 to 0.63 gives the identical answer, so the value below sits in the middle
of a wide basin rather than on the edge of a cliff.

BOUNDING THE SHOTS

Not with `peak_widths`. That descends a share of the PROMINENCE and runs
outward until the signal crosses it; on practice footage the player walks
between reps holding the ball, the signal never returns that low, and windows
of 25 s were measured for 2 s shots. Descending a share of this peak's own rise
instead gives a level the signal must have crossed, because the peak rose from
it. Windows then measured 1.9-2.4 s with a mean IoU of 0.84 against the
frame-exact truth.

WHAT THIS DOES NOT DO

It cannot tell a shot from a catch: both are the hand going up and coming down,
and the body alone does not distinguish them. That needs the ball, asked only
at each peak. And a shot already in progress when recording starts has no rise
from rest, so it has no prominence and is not found -- measured on the opening
shot of both video8 and video9.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# EMPIRICAL / PROJECT-CALIBRATED. Movement-plausibility choices measured
# against labelled footage, not published norms.

# Middle of the 0.45-0.63 basin measured on video8.
MIN_PROMINENCE = 0.55

# A clip no longer than this holds at most one attempt, so it is treated as a
# single-shot upload rather than a session to be searched.
#
# The distinction is not cosmetic. Prominence measures a peak against the
# valleys on BOTH sides, which is exactly what separates a shot from the
# fidgeting around it -- and exactly what a tightly cut clip does not have. Cut
# at the shot, one valley is missing, the prominence collapses, and the clip
# reports no shot at all. Measured: 7 of 13 single-shot clips found nothing at
# 0.55, including every clip cut close to the release.
#
# So on a short clip the question changes. It is no longer "which of these
# bumps is the shot" -- there is only one candidate -- it is "is this a shot at
# all", and that deserves a lower bar, applied to the single best peak.
SINGLE_SHOT_MAX_S = 8.0

# The bar for that one candidate. Still a real bar: a clip of someone walking
# past must find nothing, or the pipeline would score anything handed to it.
SINGLE_SHOT_MIN_PROMINENCE = 0.25

# Nobody shoots twice inside a second. Of two peaks closer than this, the more
# prominent survives. This alone replaces the old refractory timer.
MIN_GAP_S = 1.0

# Where to cut, as a share of the peak's rise above its own local floor. Low
# enough to be clear of the drive, which is a fast monotonic climb, so the
# exact value is not delicate.
WINDOW_FRACTION = 0.25

# The hand is "down" at about hip height, which is where `wrist_height_ratio`
# is near zero by construction. A little above zero, because a hand hanging at
# the side still swings.
#
# This exists because the relative cut alone is not scale-free in the way it
# looks. A shorter peak produces a HIGHER cut -- 25% of 0.59 is 0.17 -- and on
# footage where the player swings their arms walking back with the ball, the
# signal bumps above 0.17 a few frames outside the shot and the window closes
# there. Measured: two of five attempts in salah_video bounded at 0.63 s and
# 0.73 s, capturing the release and nothing else, so the report had no knee
# bend and no landing to talk about.
#
# Taking whichever cut is LOWER means the window keeps opening until the hand
# is genuinely down, and a peak's own shape can only ever widen it.
WRIST_DOWN_RATIO = 0.10

# How far outside the lift the window may extend to pick up the knee dip and
# the landing. The hand is low through both -- and also through the entire walk
# to fetch the ball, which is why this is capped.
WINDOW_PAD_S = 1.2

# A shot cycle -- dip, lift, release, finish, land -- does not happen in under
# this. Where the signal is noisy around the peak (an arm swinging while the
# player walks back with the ball produces small bumps), the cuts above can
# close on a passing dip and bound a shot at 0.6 s, which contains the release
# and nothing else. The report then has no knee bend and no landing to speak
# about, and the score is computed from a fraction of the rules.
#
# This is a statement about human movement, not a fit to any clip, and it can
# only ever widen a window -- never narrow one.
MIN_WINDOW_S = 1.5


def _local_peaks(signal: Sequence[float]) -> List[int]:
    return [
        i
        for i in range(1, len(signal) - 1)
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
    ]


def _prominence(signal: Sequence[float], peak: int) -> float:
    """Drop required before a higher point can be reached, either side.

    The definition, computed directly: walk outward until the signal rises
    above the peak, remembering the lowest point passed on each side. The
    prominence is the peak minus the HIGHER of those two valleys -- the higher
    one, because escaping to a taller peak requires crossing both.
    """
    n = len(signal)
    left_min = signal[peak]
    i = peak - 1
    while i >= 0 and signal[i] <= signal[peak]:
        left_min = min(left_min, signal[i])
        i -= 1
    right_min = signal[peak]
    j = peak + 1
    while j < n and signal[j] <= signal[peak]:
        right_min = min(right_min, signal[j])
        j += 1
    return signal[peak] - max(left_min, right_min)


def find_peaks(
    signal: Sequence[float],
    min_prominence: float = MIN_PROMINENCE,
    min_gap_frames: int = 1,
) -> List[int]:
    """Indices of shot candidates, most prominent first when they collide.

    scipy.signal.find_peaks does the same thing and is used by the probe
    script; this is implemented directly so the production path carries no
    dependency the rest of the pipeline does not already need.
    """
    scored = [
        (i, _prominence(signal, i))
        for i in _local_peaks(signal)
    ]
    scored = [(i, p) for i, p in scored if p >= min_prominence]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    kept: List[int] = []
    for index, _ in scored:
        if all(abs(index - other) >= max(1, min_gap_frames) for other in kept):
            kept.append(index)
    return sorted(kept)


def window_for(
    signal: Sequence[float], peak: int, pad_frames: int, min_frames: int = 0
) -> Tuple[int, int]:
    """Frame bounds of the attempt around `peak`."""
    n = len(signal)
    lo_bound = max(0, peak - pad_frames)
    hi_bound = min(n - 1, peak + pad_frames)

    left_floor = min(signal[lo_bound:peak + 1])
    right_floor = min(signal[peak:hi_bound + 1])
    left_cut = min(
        left_floor + WINDOW_FRACTION * (signal[peak] - left_floor),
        max(left_floor, WRIST_DOWN_RATIO),
    )
    right_cut = min(
        right_floor + WINDOW_FRACTION * (signal[peak] - right_floor),
        max(right_floor, WRIST_DOWN_RATIO),
    )

    start = peak
    while start > lo_bound and signal[start] > left_cut:
        start -= 1
    end = peak
    while end < hi_bound and signal[end] > right_cut:
        end += 1

    # Reach outward through the frames where the hand is already down: the knee
    # dip before the lift and the landing after the finish both live there, and
    # a coaching report is about them as much as about the release.
    while start > lo_bound and signal[start - 1] <= left_cut:
        start -= 1
    while end < hi_bound and signal[end + 1] <= right_cut:
        end += 1

    # Grow outward, alternating sides, until the window is at least a shot
    # long. Alternating keeps the peak near the middle rather than pushing the
    # whole window to whichever side happened to have room.
    while (end - start) < min_frames and (start > lo_bound or end < hi_bound):
        if start > lo_bound and (end - peak) >= (peak - start):
            start -= 1
        elif end < hi_bound:
            end += 1
        elif start > lo_bound:
            start -= 1
        else:
            break
    return start, end


def interpolate(values: Sequence[Optional[float]]) -> List[float]:
    """Bridge frames where the wrist was not seen, without inventing a height.

    A gap becomes a straight line between its known ends, which states
    ignorance rather than measuring. Leading and trailing gaps take the nearest
    known value; zero would read as "wrist exactly at hip height" -- the same
    conflation as the `or 0.0` bug -- and would manufacture a peak edge.
    """
    known = [i for i, v in enumerate(values) if v is not None]
    if not known:
        return [0.0] * len(values)

    out: List[float] = []
    for i, value in enumerate(values):
        if value is not None:
            out.append(float(value))
            continue
        before = [k for k in known if k < i]
        after = [k for k in known if k > i]
        if not before:
            out.append(float(values[after[0]]))
        elif not after:
            out.append(float(values[before[-1]]))
        else:
            lo, hi = before[-1], after[0]
            span = hi - lo
            weight = (i - lo) / span
            out.append(
                float(values[lo]) * (1.0 - weight) + float(values[hi]) * weight
            )
    return out


# The fastest the shooting hand can travel, in body-heights per SECOND.
#
# A hand crosses roughly one body height in a shooting motion of about half a
# second, so 4 per second is several times any real movement -- this is a
# physical impossibility bound, not a smoothness filter, and it must stay loose
# enough that no genuine release trips it.
_MAX_WRIST_RATE = 4.0

# How far below the hip the shooting hand can credibly be, in body-heights.
# Arms are not that long; a reading past this is a reconstruction failure, not
# a posture.
_MIN_CREDIBLE_WRIST = -0.75


def _reject_impossible(
    values: Sequence[Optional[float]], fps: float
) -> List[Optional[float]]:
    """Drop readings no body could produce, BEFORE prominence is measured.

    WHY SEGMENTATION NEEDS THIS AND WHY IT DID NOT HAVE IT
    ------------------------------------------------------
    Two other places in this project already guard against a single corrupt
    frame -- `phase_refiner._mask_impossible` before the knee argmin, and
    `scorer._despike` before rule aggregation. The segmentation signal had no
    such guard at all: `interpolate` fills gaps where a value is MISSING, and
    says nothing about a value that is present and absurd.

    That gap is not theoretical. Switching the pose model from `lite` to `full`
    put one frame of salah_video at t=25.17s at -1.169 -- the shooting wrist
    reported more than a full body height BELOW the hip -- with its neighbours
    at -0.41 and -0.17. `lite` produced no reading below -0.5 anywhere in the
    same clip.

    ONE FRAME WAS ENOUGH TO INVENT A SHOT. Prominence measures a peak against
    the terrain around it, so a spuriously deep trough anywhere raises the
    prominence of every ordinary bump measured across it. A hand-raise at
    t=43s, correctly ignored under `lite`, cleared MIN_PROMINENCE under `full`
    and was reported as a sixth attempt in a five-shot clip.

    The test is the same shape as `_despike`'s and for the same reason: a plain
    rate check passes a spike that follows a tracking gap, because across a
    long gap a hand really can travel far. What gives a corrupt frame away is
    that its neighbours agree with each other and only it disagrees, so the
    excursion is measured from the line joining them and bounded by the SHORTER
    side -- the value had to depart and return within it.

    Rejected samples become None, which `interpolate` then bridges. They are
    never replaced with a number, so a frame nobody could read stays unread.
    """
    n = len(values)
    if n < 3 or fps <= 0:
        return list(values)

    dt = 1.0 / fps
    out = list(values)
    for i in range(n):
        here = values[i]
        if here is None:
            continue
        if here < _MIN_CREDIBLE_WRIST:
            out[i] = None
            continue
        if i == 0 or i == n - 1:
            continue                      # nothing to corroborate an edge with
        before, after = values[i - 1], values[i + 1]
        if before is None or after is None:
            continue
        expected = (before + after) / 2.0
        if abs(here - expected) > _MAX_WRIST_RATE * dt:
            out[i] = None
    return out


def segment(
    wrist_height_ratio: Sequence[Optional[float]],
    fps: float,
    min_prominence: float = MIN_PROMINENCE,
) -> List[Tuple[int, int, int]]:
    """Split a whole video into attempts.

    Returns `(start, peak, end)` frame indices, in time order. The peak is
    carried because it is the shooting event -- what the elevation measurement
    and any ball verification are anchored on -- and recovering it from the
    window afterwards would just repeat this work.
    """
    if not wrist_height_ratio or fps <= 0:
        return []
    signal = interpolate(_reject_impossible(wrist_height_ratio, fps))
    if len(signal) < 3:
        return []

    gap = max(1, int(round(fps * MIN_GAP_S)))
    pad = max(1, int(round(fps * WINDOW_PAD_S)))
    minimum = max(1, int(round(fps * MIN_WINDOW_S)))

    peaks = find_peaks(signal, min_prominence, gap)

    # Short clip, nothing found: ask the easier question instead. Only the
    # single best candidate is considered, so this can never turn one clip into
    # several -- it can only rescue the one attempt the clip was cut around.
    if not peaks and (len(signal) / fps) <= SINGLE_SHOT_MAX_S:
        relaxed = find_peaks(signal, SINGLE_SHOT_MIN_PROMINENCE, gap)
        if relaxed:
            peaks = [max(relaxed, key=lambda i: _prominence(signal, i))]

    out = []
    for peak in peaks:
        start, end = window_for(signal, peak, pad, minimum)
        out.append((start, peak, end))
    return out


# How much of its own peak height the hand may already have at frame zero
# before the clip is judged to start mid-shot. Measured on the two clips that
# fail this way: the hand opens at 81% and 93% of its eventual peak, against
# 26-39% on clips that segment correctly.
_ALREADY_UP_FRACTION = 0.60


def explain_absence(
    wrist_height_ratio: Sequence[Optional[float]],
    fps: float,
) -> str:
    """Why no shot was found. Never called unless `segment` returned nothing.

    "No shot was found" is true and useless. Every cause below is a different
    thing for the person filming to do next, and each is separable from the
    signal we already have, so reporting them costs nothing but saying so.

    Ordered most specific first: a clip can be both short AND cut mid-shot, and
    the mid-shot diagnosis is the more actionable of the two.
    """
    values = [v for v in (wrist_height_ratio or []) if v is not None]
    if not values:
        return ("No pose was tracked. The player needs to be fully in frame, "
                "one person at a time.")

    seen = len(values) / max(1, len(wrist_height_ratio))
    if seen < 0.5:
        return (f"The player was only tracked in {seen * 100:.0f}% of frames. "
                "Check lighting, and that the whole body stays in shot.")

    signal = interpolate(wrist_height_ratio)
    peak_value = max(signal)
    rise = peak_value - min(signal)

    if peak_value > 0 and signal[0] >= _ALREADY_UP_FRACTION * peak_value:
        return (
            "The clip starts with the shooting hand already raised "
            f"({signal[0] / peak_value * 100:.0f}% of the way to its highest "
            "point), so the shot was already under way when recording began. "
            "A shot is found by how far the hand rises above where it rested "
            "beforehand, and those frames are not in this clip. Start "
            "recording about a second earlier, with the player standing still."
        )

    if rise < SINGLE_SHOT_MIN_PROMINENCE:
        return (
            f"The shooting hand only rose {rise:.2f} body heights in this clip, "
            "which is less than a shooting motion. If a shot was attempted, "
            "the hand was probably out of frame or hidden at its highest point."
        )

    duration = len(signal) / fps if fps > 0 else 0.0
    if duration > SINGLE_SHOT_MAX_S:
        return (
            "The hand rose and fell, but never far enough above the "
            "surrounding movement to stand out as a shot. On a long clip this "
            "usually means the player was moving with the ball throughout. Try "
            "a shorter clip around a single attempt."
        )

    return (
        "The hand rose and fell, but not sharply enough to be separated from "
        "the rest of the movement. The most common cause is a clip that starts "
        "too close to the shot."
    )
