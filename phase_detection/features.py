"""
Kinematic feature extraction for phase detection.

Computes velocities and positions from world landmarks and angles
using the two most recent frames in the buffer.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

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
    ankle_velocity_y: float = 0.0
    ankle_baseline_y: float = 0.0
    knee_angle: Optional[float] = None
    knee_angle_delta: float = 0.0
    hip_y_avg: float = 0.0
    hip_velocity_y: float = 0.0
    elbow_angle: Optional[float] = None
    index_y: float = 0.0
    index_velocity_y: float = 0.0
    index_align_angle: Optional[float] = None
    shoulder_y: float = 0.0
    nose_y: float = 0.0
    nose_velocity_y: float = 0.0
    total_velocity: float = 0.0
    shooting_side: str = "right"


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

    wrist_vel = ankle_vel = hip_vel = nose_vel = index_vel = 0.0
    knee_delta = 0.0

    if prev_world is not None and dt_s > 0:
        prev_wrist = _lm_y(prev_world, wrist_key)
        prev_index = _lm_y(prev_world, index_key)
        prev_ankle = _avg_y(prev_world, ("left_ankle", "right_ankle"))
        prev_hip = _avg_y(prev_world, ("left_hip", "right_hip"))
        prev_nose = _lm_y(prev_world, "nose")

        if prev_wrist is not None:
            wrist_vel = (wrist_y - prev_wrist) / dt_s
        if index_y_value is not None and prev_index is not None:
            index_vel = (index_y - prev_index) / dt_s
        if prev_ankle is not None:
            ankle_vel = (ankle_y - prev_ankle) / dt_s
        if prev_hip is not None:
            hip_vel = (hip_y - prev_hip) / dt_s
        if prev_nose is not None:
            nose_vel = (nose_y - prev_nose) / dt_s

    if prev_angles and knee_key in prev_angles and knee_angle is not None:
        prev_knee = prev_angles[knee_key]
        if prev_knee.is_valid and prev_knee.degrees is not None:
            knee_delta = knee_angle - prev_knee.degrees

    total_vel = _landmark_speed(world_landmarks, prev_world, dt_s) if prev_world else 0.0

    return KinematicFeatures(
        wrist_y=wrist_y,
        wrist_velocity_y=wrist_vel,
        ankle_y_avg=ankle_y,
        ankle_velocity_y=ankle_vel,
        ankle_baseline_y=ankle_baseline_y,
        knee_angle=knee_angle,
        knee_angle_delta=knee_delta,
        hip_y_avg=hip_y,
        hip_velocity_y=hip_vel,
        elbow_angle=elbow_angle,
        index_y=index_y,
        index_velocity_y=index_vel,
        index_align_angle=index_align_angle,
        shoulder_y=shoulder_y,
        nose_y=nose_y,
        nose_velocity_y=nose_vel,
        total_velocity=total_vel,
        shooting_side=shooting_side,
    )


def update_ankle_baseline(current_baseline: float, ankle_y: float, total_velocity: float, still_threshold: float) -> float:
    """Track standing ankle height when the player is relatively still."""
    if total_velocity < still_threshold:
        if current_baseline == 0.0:
            return ankle_y
        return 0.9 * current_baseline + 0.1 * ankle_y
    return current_baseline
