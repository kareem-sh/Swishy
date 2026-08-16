"""
Kinematic feature extraction for phase detection.

Computes velocities and positions from world landmarks and angles
using the two most recent frames in the buffer.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Guards for the image-space whole-body signals. Both are protection against
# tracking artefacts and camera cuts, not tuning knobs.
MIN_BODY_PIXEL_HEIGHT = 0.10   # player must fill >=10% of frame height
MAX_BODY_RISE_RATIO = 0.50     # nobody jumps half their own height

_MOTION_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index",
    "right_index",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class KinematicFeatures:
    """Per-frame kinematic signals used by the phase FSM."""

    wrist_y: float = 0.0
    wrist_velocity_y: float = 0.0
    ankle_y_avg: float = 0.0
    ankle_baseline_y: float = 0.0
    knee_angle: Optional[float] = None
    knee_angle_delta: float = 0.0
    hip_y_avg: float = 0.0
    elbow_angle: Optional[float] = None
    index_y: float = 0.0
    index_velocity_y: float = 0.0
    index_align_angle: Optional[float] = None
    shoulder_y: float = 0.0
    nose_y: float = 0.0
    nose_velocity_y: float = 0.0
    total_velocity: float = 0.0
    shooting_side: str = "right"
    # ---------------- IMAGE-SPACE signals (whole-body translation) ----------
    #
    # World landmarks are HIP-CENTRED: the origin is the hip midpoint. That
    # makes them ideal for joint angles and useless for whole-body motion --
    # when a player jumps, hips and ankles rise together, so nothing changes
    # in hip-relative coordinates. Measured consequence: a real jump shot
    # showed only 0.030 m of "ankle rise" in world space.
    #
    # Whole-body translation therefore has to come from image space. Both
    # signals below are divided by the player's own on-screen height, so they
    # are fractions of body height and survive changes in zoom and distance.
    #
    # body_rise_ratio: positive means the body moved UP the frame.
    # hip_x_ratio:     horizontal hip position, for stationary-vs-driving.
    body_rise_ratio: float = 0.0
    hip_x_ratio: float = 0.0
    body_pixel_height: float = 0.0

    # The RAW ankle height in image space, before any baseline is subtracted.
    #
    # `body_rise_ratio` above already answers "how high are the feet?", but it
    # answers it against a baseline that adapts frame by frame, and that choice
    # is baked in by the time anyone downstream sees the number. Shot typing
    # needs a different baseline -- one taken locally, just before this
    # attempt -- because the floor's height IN THE IMAGE depends on where the
    # player is standing, so a single figure for a whole clip is only valid
    # while the player never moves.
    #
    # Keeping the raw measurement alongside the derived one lets each consumer
    # pick its own reference. None means the ankles were not seen; it is never
    # 0.0, which would read as "ankles at the top of the frame".
    ankle_image_y: Optional[float] = None
    # Hip midpoint in normalized image coordinates. Unlike MediaPipe world
    # coordinates, this retains whole-body vertical translation and is used
    # after the shot to locate the player's jump apex.
    hip_image_y: Optional[float] = None

    # Wrist height, also from image space, as a fraction of on-screen body
    # height above the hip line. Positive means above the hips.
    #
    # The world-space `wrist_y` above is gated on MediaPipe visibility, and
    # when the gate rejects the landmark the value silently becomes 0.0 --
    # which in hip-centred coordinates reads as "the wrist is exactly at hip
    # height", a perfectly plausible number. On footage shot from behind, the
    # shooting wrist failed the gate on every frame of a clip, so the FSM saw
    # a motionless wrist at hip height for an entire jump shot and never
    # started. Image landmarks survive that gate, so this signal exists
    # whenever a pose exists at all.
    wrist_height_ratio: Optional[float] = None
    wrist_height_velocity: float = 0.0
    wrist_above_shoulder_ratio: Optional[float] = None
    # Defaults True: `extract_features` always sets this from the real
    # landmark, so the default only applies to features built by hand, where
    # an explicitly supplied `wrist_y` is by definition a real measurement.
    wrist_world_valid: bool = True

    # Distance from the centre of the visually observed ball to the guide
    # wrist, divided by the player's nose-to-ankle height in the image. This
    # is the observable part of "use the guide hand": close during the lift,
    # then clear of the ball at release. None means either the ball, guide
    # wrist, or a credible body scale was not visible; it must never be turned
    # into a zero-distance success.
    guide_hand_ball_distance_ratio: Optional[float] = None
    # 2D elbow-wrist-index angle of the non-shooting hand. This is deliberately
    # diagnostic-only: it can expose a possible wrist/thumb flick, but cannot
    # measure force and is not reliable enough to score.
    guide_wrist_align_angle: Optional[float] = None


def _lm_y(world: Dict[str, dict], name: str) -> Optional[float]:
    lm = world.get(name)
    if lm is None or not lm.get("is_reliable", False):
        return None
    return float(lm["position"][1])


def _avg_y(world: Dict[str, dict], names: tuple) -> Optional[float]:
    vals = [_lm_y(world, n) for n in names]
    valid = [v for v in vals if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _image_metrics(image_landmarks: Optional[Dict[str, dict]]) -> tuple:
    """Return ankle y, hip x/y, and body height from image space.

    Image y grows DOWNWARD, so callers must invert it to express "up".
    body_height_norm is nose-to-ankle in normalized units; dividing by it makes
    the other two invariant to zoom and camera distance.
    """
    if not image_landmarks:
        return None, None, None, 0.0

    def norm(name, axis):
        lm = image_landmarks.get(name)
        if lm is None:
            return None
        if float(lm.get("visibility", 0.0)) < 0.5:
            return None
        return float(lm["x_norm"] if axis == 0 else lm["y_norm"])

    ankles = [v for v in (norm("left_ankle", 1), norm("right_ankle", 1)) if v is not None]
    hips_x = [v for v in (norm("left_hip", 0), norm("right_hip", 0)) if v is not None]
    hips_y = [v for v in (norm("left_hip", 1), norm("right_hip", 1)) if v is not None]
    nose_y = norm("nose", 1)

    ankle_y = float(np.mean(ankles)) if ankles else None
    hip_x = float(np.mean(hips_x)) if hips_x else None
    hip_y = float(np.mean(hips_y)) if hips_y else None

    body_height = 0.0
    if ankle_y is not None and nose_y is not None:
        body_height = abs(ankle_y - nose_y)

    return ankle_y, hip_x, hip_y, body_height


def _image_wrist_ratios(
    image_landmarks: Optional[Dict[str, dict]],
    shooting_side: str,
    body_height_norm: float,
) -> tuple:
    """Wrist height above hips and above shoulder, in body-height units.

    Image y grows downward, so "above" is the smaller y -- hence reference
    minus wrist. Returns (above_hip, above_shoulder), either possibly None.

    Uses whichever wrist is visible, preferring the shooting side. A wrist the
    model cannot see at all yields None, never a fabricated number.
    """
    if not image_landmarks or body_height_norm < MIN_BODY_PIXEL_HEIGHT:
        return None, None

    def y_of(name):
        lm = image_landmarks.get(name)
        if lm is None or float(lm.get("visibility", 0.0)) < 0.3:
            return None
        return float(lm["y_norm"])

    wrist = y_of(f"{shooting_side}_wrist")
    if wrist is None:
        other = "left" if shooting_side == "right" else "right"
        wrist = y_of(f"{other}_wrist")
    if wrist is None:
        return None, None

    hips = [v for v in (y_of("left_hip"), y_of("right_hip")) if v is not None]
    shoulders = [
        v for v in (y_of("left_shoulder"), y_of("right_shoulder")) if v is not None
    ]

    above_hip = (
        (float(np.mean(hips)) - wrist) / body_height_norm if hips else None
    )
    above_shoulder = (
        (float(np.mean(shoulders)) - wrist) / body_height_norm if shoulders else None
    )
    return above_hip, above_shoulder


def _guide_hand_ball_distance_ratio(
    image_landmarks: Optional[Dict[str, dict]],
    shooting_side: str,
    body_height_norm: float,
    ball_center_xy: Optional[Tuple[float, float]],
    image_height_px: Optional[float],
) -> Optional[float]:
    """Guide-wrist distance to the ball centre, in player-height units.

    Pixel coordinates are used before normalising. Computing Euclidean
    distance directly from x_norm/y_norm would distort horizontal distance on
    every non-square video.
    """
    if (
        not image_landmarks
        or ball_center_xy is None
        or image_height_px is None
        or image_height_px <= 0.0
        or body_height_norm < MIN_BODY_PIXEL_HEIGHT
    ):
        return None

    guide_side = "left" if shooting_side == "right" else "right"
    wrist = image_landmarks.get(f"{guide_side}_wrist")
    if wrist is None:
        return None
    if (
        float(wrist.get("visibility", 0.0)) < 0.3
        or float(wrist.get("presence", 0.0)) < 0.3
    ):
        return None

    wrist_x = wrist.get("x")
    wrist_y = wrist.get("y")
    if wrist_x is None or wrist_y is None:
        return None

    body_height_px = body_height_norm * float(image_height_px)
    if body_height_px <= 0.0:
        return None
    return float(
        np.hypot(
            float(wrist_x) - float(ball_center_xy[0]),
            float(wrist_y) - float(ball_center_xy[1]),
        )
        / body_height_px
    )


def _guide_wrist_align_angle(
    image_landmarks: Optional[Dict[str, dict]],
    shooting_side: str,
) -> Optional[float]:
    """Return the guide elbow-wrist-index angle in the image plane."""
    if not image_landmarks:
        return None
    guide_side = "left" if shooting_side == "right" else "right"
    points = []
    for suffix in ("elbow", "wrist", "index"):
        landmark = image_landmarks.get(f"{guide_side}_{suffix}")
        if landmark is None:
            return None
        if (
            float(landmark.get("visibility", 0.0)) < 0.3
            or float(landmark.get("presence", 0.0)) < 0.3
        ):
            return None
        x, y = landmark.get("x"), landmark.get("y")
        if x is None or y is None:
            return None
        points.append(np.asarray((float(x), float(y)), dtype=float))

    elbow, wrist, index = points
    proximal = elbow - wrist
    distal = index - wrist
    denominator = float(np.linalg.norm(proximal) * np.linalg.norm(distal))
    if denominator <= 1e-9:
        return None
    cosine = float(np.dot(proximal, distal) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _landmark_speed(world: Dict[str, dict], prev_world: Dict[str, dict], dt_s: float) -> float:
    if dt_s <= 0:
        return 0.0
    total = 0.0
    count = 0
    # Keep the original phase signal stable when diagnostic landmarks are added.
    for name in _MOTION_LANDMARKS:
        if name not in prev_world:
            continue
        if name not in world:
            continue
        a = world[name].get("position")
        b = prev_world[name].get("position")
        if a is None or b is None:
            continue
        if not world[name].get("is_reliable") or not prev_world[name].get("is_reliable"):
            continue
        disp = np.linalg.norm(np.asarray(a) - np.asarray(b))
        total += disp / dt_s
        count += 1
    return total / count if count else 0.0


def extract_features(
    world_landmarks: Dict[str, dict],
    angles: dict,
    shooting_side: str,
    prev_world: Optional[Dict[str, dict]] = None,
    prev_angles: Optional[dict] = None,
    dt_s: float = 1 / 30.0,
    ankle_baseline_y: float = 0.0,
    image_landmarks: Optional[Dict[str, dict]] = None,
    ankle_image_baseline: float = 0.0,
    prev_image_landmarks: Optional[Dict[str, dict]] = None,
    ball_center_xy: Optional[Tuple[float, float]] = None,
    image_height_px: Optional[float] = None,
) -> KinematicFeatures:
    """Build kinematic features for the current frame."""
    wrist_key = f"{shooting_side}_wrist"
    index_key = f"{shooting_side}_index"
    knee_key = f"{shooting_side}_knee"
    elbow_key = f"{shooting_side}_elbow"
    index_align_key = f"{shooting_side}_index_align"
    shoulder_key = f"{shooting_side}_shoulder"

    wrist_y = _lm_y(world_landmarks, wrist_key) or 0.0
    index_y_value = _lm_y(world_landmarks, index_key)
    index_y = index_y_value if index_y_value is not None else 0.0
    ankle_y = _avg_y(world_landmarks, ("left_ankle", "right_ankle")) or 0.0
    hip_y = _avg_y(world_landmarks, ("left_hip", "right_hip")) or 0.0
    shoulder_y = _lm_y(world_landmarks, shoulder_key) or 0.0
    nose_y = _lm_y(world_landmarks, "nose") or 0.0

    knee_angle = None
    elbow_angle = None
    index_align_angle = None
    if knee_key in angles and angles[knee_key].is_valid:
        knee_angle = angles[knee_key].degrees
    if elbow_key in angles and angles[elbow_key].is_valid:
        elbow_angle = angles[elbow_key].degrees
    if index_align_key in angles and angles[index_align_key].is_valid:
        index_align_angle = angles[index_align_key].degrees

    # `hip_velocity_y` and `ankle_velocity_y` were REMOVED, not just unused.
    #
    # hip_velocity_y was arithmetically incapable of being anything but zero:
    # MediaPipe world landmarks are hip-centred, so the hip midpoint IS the
    # origin. Subtracting one origin from another and dividing by dt is an
    # elaborate way to compute 0.0, once per frame, forever. A threshold that
    # tested it could never fire, which is how it survived unnoticed.
    #
    # ankle_velocity_y was real arithmetic with no reader: nothing in the
    # detector, the rules or the refiner ever consulted it. Whole-body vertical
    # motion is measured in image space (`body_rise_ratio`), because hip-centred
    # world coordinates are blind to it -- so an ankle velocity in world space
    # describes the legs folding, which no rule asks about.
    #
    # Both were computed on every frame of every video.
    wrist_vel = nose_vel = index_vel = 0.0
    knee_delta = 0.0

    if prev_world is not None and dt_s > 0:
        prev_wrist = _lm_y(prev_world, wrist_key)
        prev_index = _lm_y(prev_world, index_key)
        prev_nose = _lm_y(prev_world, "nose")

        if prev_wrist is not None:
            wrist_vel = (wrist_y - prev_wrist) / dt_s
        if index_y_value is not None and prev_index is not None:
            index_vel = (index_y - prev_index) / dt_s
        if prev_nose is not None:
            nose_vel = (nose_y - prev_nose) / dt_s

    if prev_angles and knee_key in prev_angles and knee_angle is not None:
        prev_knee = prev_angles[knee_key]
        if prev_knee.is_valid and prev_knee.degrees is not None:
            knee_delta = knee_angle - prev_knee.degrees

    total_vel = _landmark_speed(world_landmarks, prev_world, dt_s) if prev_world else 0.0

    # Whole-body vertical translation, from image space, as a fraction of the
    # player's own on-screen height. Image y grows downward, so a body moving
    # UP produces a SMALLER ankle y -- hence baseline minus current.
    ankle_img_y, hip_x_norm, hip_img_y, body_height_norm = _image_metrics(
        image_landmarks
    )
    body_rise_ratio = 0.0
    if (
        ankle_img_y is not None
        and ankle_image_baseline > 0.0
        # A player occupying less than a tenth of the frame height is either
        # mis-detected or too far away to measure. Without this guard the
        # division blows up: an observed value of 98.2 body-heights came from a
        # near-zero denominator.
        and body_height_norm >= MIN_BODY_PIXEL_HEIGHT
    ):
        raw_ratio = (ankle_image_baseline - ankle_img_y) / body_height_norm
        # Nobody jumps their own height. Anything beyond this is a tracking
        # artefact or a camera cut, not motion, so it is clamped rather than
        # trusted.
        body_rise_ratio = max(-MAX_BODY_RISE_RATIO, min(MAX_BODY_RISE_RATIO, raw_ratio))
    hip_x_ratio = hip_x_norm if hip_x_norm is not None else 0.0

    wrist_above_hip, wrist_above_shoulder = _image_wrist_ratios(
        image_landmarks, shooting_side, body_height_norm
    )
    guide_ball_distance = _guide_hand_ball_distance_ratio(
        image_landmarks,
        shooting_side,
        body_height_norm,
        ball_center_xy,
        image_height_px,
    )
    guide_wrist_angle = _guide_wrist_align_angle(
        image_landmarks,
        shooting_side,
    )
    wrist_ratio_velocity = 0.0
    if wrist_above_hip is not None and prev_image_landmarks is not None and dt_s > 0:
        _, _, _, prev_body_height = _image_metrics(prev_image_landmarks)
        prev_above_hip, _ = _image_wrist_ratios(
            prev_image_landmarks, shooting_side, prev_body_height
        )
        if prev_above_hip is not None:
            wrist_ratio_velocity = (wrist_above_hip - prev_above_hip) / dt_s

    return KinematicFeatures(
        wrist_y=wrist_y,
        wrist_velocity_y=wrist_vel,
        ankle_y_avg=ankle_y,
        ankle_baseline_y=ankle_baseline_y,
        knee_angle=knee_angle,
        knee_angle_delta=knee_delta,
        hip_y_avg=hip_y,
        elbow_angle=elbow_angle,
        index_y=index_y,
        index_velocity_y=index_vel,
        index_align_angle=index_align_angle,
        shoulder_y=shoulder_y,
        nose_y=nose_y,
        nose_velocity_y=nose_vel,
        total_velocity=total_vel,
        shooting_side=shooting_side,
        body_rise_ratio=body_rise_ratio,
        hip_x_ratio=hip_x_ratio,
        body_pixel_height=body_height_norm,
        ankle_image_y=ankle_img_y,
        hip_image_y=hip_img_y,
        wrist_height_ratio=wrist_above_hip,
        wrist_height_velocity=wrist_ratio_velocity,
        wrist_above_shoulder_ratio=wrist_above_shoulder,
        wrist_world_valid=_lm_y(world_landmarks, wrist_key) is not None,
        guide_hand_ball_distance_ratio=guide_ball_distance,
        guide_wrist_align_angle=guide_wrist_angle,
    )


def update_ankle_image_baseline(
    current_baseline: float,
    image_landmarks: Optional[Dict[str, dict]],
    total_velocity: float,
    still_threshold: float,
) -> float:
    """Track where the feet sit on screen while the player is standing still.

    This is the ground reference for whole-body vertical motion. It is only
    updated while the player is relatively still, so a jump cannot drag the
    baseline up with it.
    """
    ankle_y, _, _, body_height = _image_metrics(image_landmarks)
    if ankle_y is None:
        return current_baseline

    # Ignore frames where the detected body is implausibly small. These are
    # partial or spurious detections, and they are common on the first frames
    # of a clip before tracking settles. Seeding the ground reference from one
    # put the baseline 0.2 body heights above the real floor and pinned the
    # rise signal at its clamp for an entire jump shot.
    if body_height < MIN_BODY_PIXEL_HEIGHT:
        return current_baseline

    if current_baseline <= 0.0:
        # Seed on the first credible observation rather than waiting for the
        # player to be still: someone already walking when the clip opens is
        # never still, and with no baseline the rise signal stays 0.0 forever,
        # which reads as "this player never left the floor".
        return ankle_y

    # Asymmetric on purpose. Image y grows downward, so a LARGER ankle y means
    # the feet are lower — and the floor is the lowest the feet ever go. A
    # reading below the current baseline is therefore evidence about where the
    # ground actually is and is adopted quickly; a reading above it is what a
    # jump looks like, so it must not drag the reference up with it.
    if ankle_y > current_baseline:
        return 0.6 * current_baseline + 0.4 * ankle_y
    if total_velocity >= still_threshold:
        return current_baseline
    return 0.9 * current_baseline + 0.1 * ankle_y


def update_ankle_baseline(
    current_baseline: float,
    ankle_y: Optional[float],
    total_velocity: float,
    still_threshold: float,
) -> float:
    """Track standing ankle height when the player is relatively still.

    `ankle_y` is Optional, and None means the ankles were not seen. It must
    not be replaced by 0.0 anywhere upstream: in hip-centred world coordinates
    0.0 means "ankles level with the hips", which is a real posture, so the
    baseline cannot tell it apart from a missing reading and would blend it in
    -- or seed itself with it outright on the first still frame.

    This is the guard `update_ankle_image_baseline` below has always had. This
    function did not, and the caller collapsed None to 0.0 before reaching it.
    """
    if ankle_y is None:
        return current_baseline
    if total_velocity < still_threshold:
        if current_baseline == 0.0:
            return ankle_y
        return 0.9 * current_baseline + 0.1 * ankle_y
    return current_baseline
