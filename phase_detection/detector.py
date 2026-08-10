"""
Finite state machine for basketball jump-shot phase detection.

Uses kinematic features + hysteresis to avoid phase flicker.
Thresholds loaded from config/phases.yaml.
"""

from typing import Optional

from phase_detection.features import KinematicFeatures
from phase_detection.phases import PHASE_LABELS, TRANSITIONS
from utils.config_loader import load_yaml


class ShotPhaseDetector:
    """Detect current shot phase from per-frame kinematic features."""

    # Where each phase is forced to go if it overstays its timeout.
    #
    # Every non-terminal phase needs an escape. When only the second half of
    # the shot had one, `knee_flexion` and `ball_lift` became sinks: on real
    # footage the FSM entered them, the single exit condition never fired, and
    # the attempt was lost entirely -- 4 of 13 shots produced no score and no
    # feedback for this reason alone.
    #
    # A timeout BEFORE the release abandons the attempt; a timeout AFTER it
    # closes the attempt out.
    #
    # The asymmetry is the important part. `release` is the event that defines
    # a shot, so a timeout must never invent one. Advancing ball_lift into
    # release on expiry did exactly that: on a 45 s clip of a player mostly
    # standing around, the FSM manufactured a complete phase cycle every few
    # seconds and the run ended with a fabricated attempt in flight. Before
    # the release the honest conclusion is "that was not a shot", so the FSM
    # returns to rest and the candidate is discarded. After the release the
    # shot is already established and only needs closing.
    _TIMEOUT_TARGET = {
        "loading": "ready_stance",
        "knee_flexion": "ready_stance",
        "ball_lift": "ready_stance",
        "jump": "ready_stance",
        "release": "follow_through",
        "follow_through": "landing",
        "landing": "ready_stance",
    }

    def __init__(self):
        cfg = load_yaml("phases.yaml")
        self._cfg = cfg
        self._thresholds = cfg.get("thresholds", {})

        # Confirmation and dwell are DURATIONS. Sample footage runs from 12 to
        # 30 fps and includes slow motion, so "2 frames" means 0.067 s on one
        # clip and 0.167 s on another -- the same config behaving differently
        # per file. Frame counts remain as a fallback for callers that supply
        # no timestamps.
        self._hysteresis_s = float(cfg.get("hysteresis_s", 0.07))
        self._min_dwell_s = float(cfg.get("min_dwell_s", 0.07))
        self._hysteresis_frames = cfg.get("hysteresis_frames", 2)
        self._min_dwell_frames = cfg.get("min_dwell_frames", 2)

        seg_cfg = load_yaml("segmentation.yaml")
        self._phase_timeouts_s = seg_cfg.get("phase_timeouts_s", {}) or {}

        self.phase = "ready_stance"
        self._pending_phase: Optional[str] = None
        self._pending_count = 0
        self._pending_since_ms: Optional[int] = None
        self._frames_in_phase = 0
        self._phase_entered_ms: Optional[int] = None
        self._last_progress_ms: Optional[int] = None
        self._progress_value: Optional[float] = None
        self._wrist_peak_y = 0.0
        self._knee_min_angle = 180.0
        self._in_shot = False
        self.timed_out_last_update = False

    def reset(self):
        self.phase = "ready_stance"
        self._pending_phase = None
        self._pending_count = 0
        self._pending_since_ms = None
        self._frames_in_phase = 0
        self._phase_entered_ms = None
        self._last_progress_ms = None
        self._progress_value = None
        self._wrist_peak_y = 0.0
        self._knee_min_angle = 180.0
        self._in_shot = False
        self.timed_out_last_update = False

    # How far the shooting hand must move for the shot to count as still
    # progressing. Body heights when the image-space signal is available,
    # metres otherwise; the two are close enough in magnitude for one value.
    _PROGRESS_EPSILON = 0.04

    def _note_progress(self, f: KinematicFeatures, timestamp_ms: Optional[int]) -> None:
        """Restart the stall clock whenever the hand is still moving.

        Timeouts measure time WITHOUT PROGRESS, not time in a phase. A fixed
        cap on time in phase cannot work across this footage: a ball lift that
        takes 0.4 s at normal speed takes 5.4 s in slow motion, so any cap
        tight enough to rescue a stuck state also aborts a real slow-motion
        shot mid-flight. Movement is the thing that distinguishes them.
        """
        if timestamp_ms is None:
            return
        value = f.wrist_height_ratio
        if value is None and f.wrist_world_valid:
            value = f.wrist_y
        if value is None:
            return
        if self._progress_value is None or abs(value - self._progress_value) >= self._PROGRESS_EPSILON:
            self._progress_value = value
            self._last_progress_ms = timestamp_ms

    def _check_timeout(self, timestamp_ms: Optional[int]) -> Optional[str]:
        """Return the phase to force into, if the current one has stalled."""
        if timestamp_ms is None or self._phase_entered_ms is None:
            return None
        limit_s = self._phase_timeouts_s.get(self.phase)
        if limit_s is None:
            return None
        since = self._phase_entered_ms
        if self._last_progress_ms is not None:
            since = max(since, self._last_progress_ms)
        if (timestamp_ms - since) / 1000.0 < float(limit_s):
            return None
        return self._TIMEOUT_TARGET.get(self.phase)

    @property
    def phase_label(self) -> str:
        return PHASE_LABELS.get(self.phase, self.phase)

    def update(
        self,
        features: KinematicFeatures,
        timestamp_ms: Optional[int] = None,
    ) -> str:
        if self._phase_entered_ms is None and timestamp_ms is not None:
            self._phase_entered_ms = timestamp_ms

        self.timed_out_last_update = False
        self._note_progress(features, timestamp_ms)
        forced = self._check_timeout(timestamp_ms)
        if forced is not None:
            self.phase = forced
            self._phase_entered_ms = timestamp_ms
            self._frames_in_phase = 0
            self._pending_phase = None
            self._pending_count = 0
            self.timed_out_last_update = True
            self._track_shot_metrics(features)
            return self.phase

        candidate = self._evaluate_transition(features)
        if candidate and candidate != self.phase:
            if candidate != self._pending_phase:
                self._pending_phase = candidate
                self._pending_count = 0
                self._pending_since_ms = timestamp_ms
            self._pending_count += 1

            if self._confirmed(timestamp_ms) and self._dwelled(timestamp_ms):
                if self._is_valid_transition(self.phase, candidate):
                    self.phase = candidate
                    self._frames_in_phase = 0
                    self._phase_entered_ms = timestamp_ms
                self._pending_phase = None
                self._pending_count = 0
                self._pending_since_ms = None
        elif candidate == self.phase:
            self._pending_phase = None
            self._pending_count = 0
            self._pending_since_ms = None
        # When candidate is None the player is holding the current phase — keep it.

        self._frames_in_phase += 1
        self._track_shot_metrics(features)
        return self.phase

    def _confirmed(self, timestamp_ms: Optional[int]) -> bool:
        """Has the candidate phase persisted long enough to be believed?"""
        if timestamp_ms is None or self._pending_since_ms is None:
            return self._pending_count >= self._hysteresis_frames
        held_s = (timestamp_ms - self._pending_since_ms) / 1000.0
        # A single frame can satisfy the duration at low frame rates; require
        # at least two observations so one noisy frame is never enough.
        return held_s >= self._hysteresis_s and self._pending_count >= 2

    def _dwelled(self, timestamp_ms: Optional[int]) -> bool:
        """Has the current phase been held long enough to be allowed to exit?"""
        if self.phase == "ready_stance":
            return True
        if timestamp_ms is None or self._phase_entered_ms is None:
            return self._frames_in_phase >= self._min_dwell_frames
        return (timestamp_ms - self._phase_entered_ms) / 1000.0 >= self._min_dwell_s

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

    # ---------------------------------------------------------- wrist signal
    # The wrist drives most of the shot. Its world-space position is gated on
    # MediaPipe visibility and, when the gate rejects it, silently reads 0.0 --
    # indistinguishable from "the wrist is at hip height" in hip-centred
    # coordinates. On footage filmed from behind that happened on every frame
    # of a clip, so the FSM watched a motionless wrist through an entire jump
    # shot. These helpers prefer world space when it was genuinely observed and
    # fall back to the image-space ratio, which survives the same gate.

    def _wrist_rising(self, f: KinematicFeatures) -> bool:
        if f.wrist_world_valid:
            return f.wrist_velocity_y > self._t("ball_lift_wrist_velocity", 0.03)
        return f.wrist_height_velocity > self._t("ball_lift_wrist_rise_rate", 0.25)

    def _wrist_falling(self, f: KinematicFeatures) -> bool:
        if f.wrist_world_valid:
            return f.wrist_velocity_y < self._t(
                "follow_through_wrist_down_velocity", -0.02
            )
        return f.wrist_height_velocity < -self._t("ball_lift_wrist_rise_rate", 0.25)

    def _wrist_in_carry_window(self, f: KinematicFeatures) -> bool:
        """Is the ball somewhere it could plausibly be shot from?"""
        if f.wrist_world_valid:
            return (
                f.hip_y_avg - self._t("carry_below_hip", 0.15)
                < f.wrist_y
                < f.shoulder_y + self._t("carry_above_shoulder", 0.30)
            )
        if f.wrist_height_ratio is None:
            return False
        # Body-height fractions: hips are 0, shoulders roughly 0.30.
        return -0.10 < f.wrist_height_ratio < 0.55

    @staticmethod
    def _wrist_above_hips(f: KinematicFeatures) -> bool:
        if f.wrist_world_valid:
            return f.wrist_y > f.hip_y_avg
        return f.wrist_height_ratio is not None and f.wrist_height_ratio > 0.0

    def _evaluate_transition(self, f: KinematicFeatures) -> Optional[str]:
        t = self._t
    
        if self.phase == "ready_stance":
            # "Is the ball in a position it could be shot from?"
            #
            # This used to be `wrist_y < hip_y_avg + 0.35`. World landmarks are
            # hip-centred, so hip_y_avg is ~0 by construction and the test
            # silently meant "the wrist is under 0.35 m above the hips". A
            # player who sets the ball at chin height is above that, so the
            # condition could never be met and the FSM sat in ready_stance for
            # the whole clip -- 2 of 13 shots were lost exactly this way.
            #
            # Both bounds are now relative to real anatomy: anywhere from just
            # below the hips to somewhat above the shoulder is a plausible
            # carry or set position.
            # NOTE: `hip_velocity_y` is deliberately not tested here. World
            # landmarks are hip-centred, so the hip midpoint IS the origin and
            # its velocity is identically zero on every frame -- the old
            # `loading_hip_drop_velocity` condition could never fire. Knee
            # flexion is the live signal for a dip.
            knee_loading = (
                f.knee_angle is not None
                and f.knee_angle_delta < t("loading_knee_flex_delta", -1.0)
            )

            if not self._wrist_in_carry_window(f):
                return None
            if self._wrist_rising(f):
                return "ball_lift"
            if knee_loading:
                return "loading"
            return None

        if self.phase == "loading":
            # The lift is checked first: once the ball is on its way up, the
            # knee is already extending, and calling that `knee_flexion`
            # diverts the shot into a state whose only exit is the very
            # condition just observed.
            if self._wrist_rising(f):
                return "ball_lift"
            if f.knee_angle is not None and f.knee_angle_delta > t("knee_extend_delta", 0.5):
                return "knee_flexion"
            return None

        if self.phase == "knee_flexion":
            if self._wrist_rising(f) and self._wrist_above_hips(f):
                return "ball_lift"
            return None

        if self.phase == "ball_lift":
            # Whole-body elevation, measured in IMAGE space as a fraction of
            # the player's own on-screen height.
            #
            # This previously read `ankle_y_avg - ankle_baseline_y` from world
            # landmarks. Those are HIP-CENTRED -- the origin is the hip midpoint
            # -- so when a player jumps, hips and ankles rise together and the
            # difference barely moves: a real jump shot measured 0.030 m. The
            # test therefore fired on noise during flat-footed set shots and
            # stayed quiet during genuine jumps, which is exactly backwards.
            if f.body_rise_ratio > t("jump_body_rise_ratio", 0.06):
                return "jump"
            wrist_over_hip = self._wrist_above_hips(f)
            if wrist_over_hip and self._wrist_falling(f):
                return "release"
            if (
                f.elbow_angle is not None
                and f.elbow_angle > t("set_shot_elbow_min", 145)
                and wrist_over_hip
                and not self._wrist_rising(f)
            ):
                return "release"
            index_release = (
                f.index_align_angle is not None
                and f.index_align_angle > t("release_index_align_min", 150)
                and f.index_velocity_y > t("release_index_up_velocity", 0.03)
            )
            if index_release and wrist_over_hip:
                return "release"
            return None

        if self.phase == "jump":
            wrist_near_peak = f.wrist_y >= self._wrist_peak_y - t("release_peak_tolerance", 0.02)
            elbow_extended = f.elbow_angle is not None and f.elbow_angle > t("release_elbow_min", 150)
            wrist_slow = abs(f.wrist_velocity_y) < t("release_wrist_velocity_max", 0.05)
            index_snap = f.index_velocity_y > t("release_index_up_velocity", 0.04)
            index_extended = (
                f.index_align_angle is not None
                and f.index_align_angle > t("release_index_align_min", 155)
            )
            if (wrist_near_peak and wrist_slow) or (elbow_extended and f.wrist_velocity_y < 0):
                return "release"
            if index_snap and elbow_extended:
                return "release"
            if index_extended and elbow_extended:
                return "release"
            return None

        if self.phase == "release":
            index_follow = (
                f.index_align_angle is not None
                and f.index_align_angle > t("follow_through_index_align_min", 160)
            )
            if f.wrist_velocity_y < t("follow_through_wrist_down_velocity", -0.02):
                return "follow_through"
            if f.elbow_angle is not None and f.elbow_angle > t("follow_through_elbow_min", 155):
                return "follow_through"
            if index_follow and f.index_velocity_y < t("follow_through_index_down_velocity", -0.02):
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
