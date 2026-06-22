"""
Finite state machine for basketball jump-shot phase detection.

Uses kinematic features + hysteresis to avoid phase flicker.
Thresholds loaded from config/phases.yaml.
"""

from typing import Dict, Optional

from phase_detection.features import KinematicFeatures
from phase_detection.phases import PHASE_LABELS, TRANSITIONS
from utils.config_loader import load_yaml


class ShotPhaseDetector:
    """Detect current shot phase from per-frame kinematic features."""

    def __init__(self):
        cfg = load_yaml("phases.yaml")
        self._cfg = cfg
        self._hysteresis_frames = cfg.get("hysteresis_frames", 3)
        self._thresholds = cfg.get("thresholds", {})

        self.phase = "ready_stance"
        self._pending_phase: Optional[str] = None
        self._pending_count = 0
        self._wrist_peak_y = 0.0
        self._knee_min_angle = 180.0
        self._in_shot = False

    def reset(self):
        self.phase = "ready_stance"
        self._pending_phase = None
        self._pending_count = 0
        self._wrist_peak_y = 0.0
        self._knee_min_angle = 180.0
        self._in_shot = False

    @property
    def phase_label(self) -> str:
        return PHASE_LABELS.get(self.phase, self.phase)

    def update(self, features: KinematicFeatures) -> str:
        candidate = self._evaluate_transition(features)
        if candidate and candidate != self.phase:
            if candidate == self._pending_phase:
                self._pending_count += 1
            else:
                self._pending_phase = candidate
                self._pending_count = 1

            if self._pending_count >= self._hysteresis_frames:
                if self._is_valid_transition(self.phase, candidate):
                    self.phase = candidate
                self._pending_phase = None
                self._pending_count = 0
        else:
            self._pending_phase = None
            self._pending_count = 0

        self._track_shot_metrics(features)
        return self.phase

    def _is_valid_transition(self, current: str, nxt: str) -> bool:
        allowed = TRANSITIONS.get(current, [])
        return nxt in allowed or nxt == "ready_stance"

    def _track_shot_metrics(self, f: KinematicFeatures):
        if f.wrist_y > self._wrist_peak_y:
            self._wrist_peak_y = f.wrist_y
        if f.knee_angle is not None and f.knee_angle < self._knee_min_angle:
            self._knee_min_angle = f.knee_angle

        if self.phase not in ("ready_stance", "landing"):
            self._in_shot = True
        if self.phase == "ready_stance" and not self._in_shot:
            self._wrist_peak_y = f.wrist_y
            self._knee_min_angle = f.knee_angle or 180.0
        if self.phase == "landing":
            self._in_shot = False

    def _t(self, key: str, default: float) -> float:
        return float(self._thresholds.get(key, default))

    def _evaluate_transition(self, f: KinematicFeatures) -> Optional[str]:
        t = self._t

        if self.phase == "ready_stance":
            if (
                f.hip_velocity_y < t("loading_hip_drop_velocity", -0.02)
                or (f.knee_angle is not None and f.knee_angle_delta < t("loading_knee_flex_delta", -1.0))
            ) and f.wrist_y < f.shoulder_y + t("loading_wrist_below_shoulder", 0.05):
                return "loading"
            return None

        if self.phase == "loading":
            if f.knee_angle is not None and f.knee_angle_delta > t("knee_extend_delta", 0.5):
                return "knee_flexion"
            if f.wrist_velocity_y > t("ball_lift_wrist_velocity", 0.03):
                return "ball_lift"
            return None

        if self.phase == "knee_flexion":
            if f.wrist_velocity_y > t("ball_lift_wrist_velocity", 0.03) and f.wrist_y > f.hip_y_avg:
                return "ball_lift"
            return None

        if self.phase == "ball_lift":
            ankle_rise = f.ankle_y_avg - f.ankle_baseline_y
            if ankle_rise > t("jump_ankle_rise", 0.03) or f.ankle_velocity_y > t("jump_ankle_velocity", 0.02):
                return "jump"
            return None

        if self.phase == "jump":
            wrist_near_peak = f.wrist_y >= self._wrist_peak_y - t("release_peak_tolerance", 0.02)
            elbow_extended = f.elbow_angle is not None and f.elbow_angle > t("release_elbow_min", 150)
            wrist_slow = abs(f.wrist_velocity_y) < t("release_wrist_velocity_max", 0.05)
            if (wrist_near_peak and wrist_slow) or (elbow_extended and f.wrist_velocity_y < 0):
                return "release"
            return None

        if self.phase == "release":
            if f.wrist_velocity_y < t("follow_through_wrist_down_velocity", -0.02):
                return "follow_through"
            if f.elbow_angle is not None and f.elbow_angle > t("follow_through_elbow_min", 155):
                return "follow_through"
            return None

        if self.phase == "follow_through":
            ankle_near_base = abs(f.ankle_y_avg - f.ankle_baseline_y) < t("landing_ankle_tolerance", 0.04)
            if ankle_near_base and f.ankle_velocity_y < t("landing_ankle_velocity_max", 0.01):
                return "landing"
            return None

        if self.phase == "landing":
            if f.total_velocity < t("ready_max_velocity", 0.15):
                return "ready_stance"
            return None

        return None
