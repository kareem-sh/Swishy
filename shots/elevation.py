"""How far the player left the floor during one attempt.

This is the measurement that separates a jump shot from a set shot. It is a
separate module from `shots/classifier.py` because the classifier decides and
this measures, and the two failed for independent reasons: the threshold was
right all along while the measurement under it was wrong.

WHY THE BASELINE IS LOCAL

The floor is not a horizontal line in a camera image. It is a plane seen in
perspective, so a player standing further from the camera has their feet
HIGHER in the frame while standing on exactly the same floor. A single
"ground height" for a whole clip is therefore only valid while the player
never moves -- and in practice footage they walk out to collect the ball
between every rep.

Measured consequence of a whole-clip baseline: on a clip where the player
retrieves the ball each time, real jump shots reported NEGATIVE elevation --
their feet read as below the floor -- because the reference had been averaged
across positions they were not standing in.

The fix is not a better global number. It is to stop having one. Each attempt
takes its reference from the player's own stance in the seconds immediately
before that attempt, which is the same spot, at the same distance, in the same
perspective. Camera geometry, player height and lens all cancel because they
are identical on both sides of the subtraction.

WHY THE ANKLE, AND NOT THE TOE

The toe is the physically better landmark: rising onto the balls of the feet
lifts the ankle without the foot leaving the ground, while the toe stays down.
Measured on 14 labelled shots, the toe is nonetheless WORSE -- it leaves
0.001 body heights between the highest set shot and the threshold, against
0.019 for the ankle. `foot_index` is among MediaPipe's least stable landmarks:
small, frequently clipped by the frame edge, often occluded. The theoretical
gain is swallowed by measurement noise. The heel behaves like the ankle and
adds nothing.

VALIDATION (see scripts/peak_probe.py to reproduce)

14 labelled attempts, 13 set shots and 1 jump shot, from video8 (frame-exact
windows in single_shot/manifest.json) and salah_video: 14/14 correct, with the
existing 0.12 threshold unchanged.

Honest limits: the margin above the highest set shot is 0.019 body heights,
which is thin, and only ONE labelled jump shot has been measured this way.
Set-shot detection is well evidenced; jump detection is not yet.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

# All durations are in SECONDS, never frame counts: sample footage runs from
# 12 to 30 fps and includes slow motion.

# How far back to look for the stance. A bound on work and on how much
# unrelated footage can be dragged in, not a definition of the stance itself.
MAX_LOOKBACK_S = 2.5

# The stance is defined by EVIDENCE, not by a clock: it is the frames before
# the shot where the shooting hand is still low. A fixed "0.8 to 2.0 seconds
# before" window seems equivalent and is not -- the single-shot clips are cut
# at the shot, so a fixed window lands entirely before frame zero and finds
# nothing, while the player is plainly standing there in the frames that do
# exist. Anchoring on the hand instead adapts to whatever was recorded, and to
# slow motion, where the same movement takes four times as long.
#
# A frame counts as stance while the wrist sits within this fraction of the way
# from its lowest point to its peak. Low enough to be clear of the drive; the
# lift is a fast, monotonic climb, so the exact fraction is not delicate.
STANCE_WRIST_FRACTION = 0.25

# How far either side of the shooting event to look for peak elevation. The
# feet are highest around release, and a window this tight keeps the walk
# before and after the attempt out of the measurement -- which is what
# corrupted it when the whole attempt window was used.
EVENT_HALF_WINDOW_S = 0.5

# Below this the detected body is too small for its own height to be a
# trustworthy divisor. Matches the guard in phase_detection/features.py.
MIN_BODY_HEIGHT = 0.10

# A stance shorter than this is not enough to establish where the floor is.
MIN_BASELINE_SAMPLES = 5


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _stance_indices(
    timestamps_ms: Sequence[int],
    wrist_height_ratio: Sequence[Optional[float]],
    event_ms: int,
) -> List[int]:
    """Frames before the shot in which the shooting hand is still down."""
    window = [
        i
        for i, ts in enumerate(timestamps_ms)
        if event_ms - int(MAX_LOOKBACK_S * 1000) <= ts < event_ms
    ]
    seen = [
        wrist_height_ratio[i] for i in window if wrist_height_ratio[i] is not None
    ]
    if not seen:
        return []
    low, high = min(seen), max(seen)
    if high <= low:
        # The hand never moved in the lookback: all of it is stance.
        cutoff = high
    else:
        cutoff = low + STANCE_WRIST_FRACTION * (high - low)
    return [
        i
        for i in window
        if wrist_height_ratio[i] is not None and wrist_height_ratio[i] <= cutoff
    ]


def takeoff_elevation(
    timestamps_ms: Sequence[int],
    ankle_image_y: Sequence[Optional[float]],
    body_height: Sequence[float],
    wrist_height_ratio: Sequence[Optional[float]],
    event_ms: int,
) -> Optional[float]:
    """Peak elevation of the feet around `event_ms`, in body heights.

    Returns None when the stance before the attempt cannot be observed -- a
    recording that opens mid-shot has no "before", and there is no honest way
    to invent one. Callers must treat None as "not measured" and decline to
    classify, never as zero: zero is a specific claim that the player stayed
    on the floor.

    Image y grows downward, so the feet rising produces a SMALLER y; hence
    baseline minus current.
    """
    if not timestamps_ms:
        return None

    usable = [
        i
        for i, (ankle, height) in enumerate(zip(ankle_image_y, body_height))
        if ankle is not None and height >= MIN_BODY_HEIGHT
    ]
    if not usable:
        return None
    usable_set = set(usable)

    stance = [
        float(ankle_image_y[i])
        for i in _stance_indices(timestamps_ms, wrist_height_ratio, event_ms)
        if i in usable_set
    ]

    event_lo = event_ms - int(EVENT_HALF_WINDOW_S * 1000)
    event_hi = event_ms + int(EVENT_HALF_WINDOW_S * 1000)
    event: List[Tuple[float, float]] = [
        (float(ankle_image_y[i]), float(body_height[i]))
        for i in usable
        if event_lo <= timestamps_ms[i] <= event_hi
    ]

    if len(stance) < MIN_BASELINE_SAMPLES or not event:
        return None

    floor = _median(stance)
    return max((floor - ankle) / height for ankle, height in event)


def shooting_event_ms(
    timestamps_ms: Sequence[int],
    wrist_height_ratio: Sequence[Optional[float]],
) -> Optional[int]:
    """When the shooting event happened: the frame the wrist was highest.

    An argmax over a captured attempt, deliberately, rather than a live
    threshold on velocity. A maximum over an array cannot get stuck partway
    and cannot fire early; a threshold can do both, and did.
    """
    best_ms: Optional[int] = None
    best: float = float("-inf")
    for ts, ratio in zip(timestamps_ms, wrist_height_ratio):
        if ratio is None:
            continue
        if ratio > best:
            best, best_ms = float(ratio), int(ts)
    return best_ms
