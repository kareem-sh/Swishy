"""Live state machine for finding shots in a stream of pose frames.

FOUR STATES, DELIBERATELY
-------------------------
This used to run the same eight phases the coaching report talks about:
ready -> loading -> knee_flexion -> ball_lift -> jump -> release ->
follow_through -> landing. That made finding a shot depend on seven
consecutive threshold gates, and any one of them failing lost the shot
entirely -- not degraded it, lost it. Measured: 4 shots found in a ten-shot
video, 0 found in a side-on jump shot that produced 7 discarded candidates.

The states that survive are the ones a shot cannot happen without:

    ready     not shooting
    rise      the shooting hand is on its way up
    release   the ball has left the hand
    recovery  the finish

Three gates instead of seven. The dip, the take-off and the landing are still
reported to the player -- they are just worked out afterwards by
feedback.phase_refiner, which has the whole shot in hand and can locate them
by extrema (the frame where the knee was most bent) instead of by live
velocity thresholds. A minimum over an array cannot get stuck.

Thresholds come from config/phases.yaml, the state graph from
config/phase_model.yaml.
"""

from typing import Optional

from phase_detection.features import KinematicFeatures
from phase_detection.phases import (
    CORE_ANCHOR,
    CORE_LABELS,
    CORE_REST,
    CORE_TRANSITIONS,
)
from utils.config_loader import load_yaml


class ShotPhaseDetector:
    """Detect the current core state from per-frame kinematic features."""

    # Where each state is forced to go if it overstays its timeout.
    #
    # The asymmetry is the important part. `release` is the event that defines
    # a shot, so a timeout must never invent one: before the ball has gone the
    # honest conclusion is "that was not a shot", so the machine returns to
    # rest and the candidate is dropped. After the ball has gone the shot is
    # already established and the timeout only closes it out.
    _TIMEOUT_TARGET = {
        "rise": CORE_REST,
        "release": "recovery",
        "recovery": CORE_REST,
    }

    # States after the ball has gone. These time out on the wall clock.
    #
    # Everywhere else the clock measures time WITHOUT PROGRESS rather than
    # time in state, because a lift that takes 0.4 s at normal speed takes
    # 5.4 s in slow motion and only movement can tell a slow shot from a stuck
    # one. Once the ball has left the hand there is no flight left to protect
    # and the movement test backfires: a player walking out to collect the
    # ball swings their arms, that reads as progress, and the clock resets
    # forever. Measured consequence of leaving it progress-based: the finish
    # ran 6.4 s against a 3.0 s limit and the machine never returned to rest.
    _ABSOLUTE_TIMEOUT_STATES = frozenset({"release", "recovery"})

    # How far the shooting hand must move for the shot to count as still
    # progressing. Body heights when the image-space signal is available,
    # metres otherwise; the two are close enough in magnitude for one value.
    _PROGRESS_EPSILON = 0.04

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
        self._state_timeouts_s = seg_cfg.get("state_timeouts_s", {}) or {}

        self.phase = CORE_REST
        self._pending_phase: Optional[str] = None
        self._pending_count = 0
        self._pending_since_ms: Optional[int] = None
        self._frames_in_phase = 0
        self._phase_entered_ms: Optional[int] = None
        self._last_progress_ms: Optional[int] = None
        self._progress_value: Optional[float] = None
        self._wrist_peak = float("-inf")
        # Floor of the hand's current upward run, and how far it has climbed
        # from that floor. See `_track_climb`.
        self._rise_floor: Optional[float] = None
        self._climb = 0.0
        # Latched within one rise. See `_ball_went_up`.
        self._went_up = False
        self._knee_min_angle = 180.0
        self.timed_out_last_update = False

    def reset(self):
        self.__init__()

    # -------------------------------------------------------------- timing
    def _note_progress(self, f: KinematicFeatures, timestamp_ms: Optional[int]) -> None:
        """Restart the stall clock whenever the hand is still moving."""
        if timestamp_ms is None:
            return
        value = self._wrist_level(f)
        if value is None:
            return
        if (
            self._progress_value is None
            or abs(value - self._progress_value) >= self._PROGRESS_EPSILON
        ):
            self._progress_value = value
            self._last_progress_ms = timestamp_ms

    def _check_timeout(self, timestamp_ms: Optional[int]) -> Optional[str]:
        """Return the state to force into, if the current one has stalled."""
        if timestamp_ms is None or self._phase_entered_ms is None:
            return None
        limit_s = self._state_timeouts_s.get(self.phase)
        if limit_s is None:
            return None
        since = self._phase_entered_ms
        if (
            self._last_progress_ms is not None
            and self.phase not in self._ABSOLUTE_TIMEOUT_STATES
        ):
            since = max(since, self._last_progress_ms)
        if (timestamp_ms - since) / 1000.0 < float(limit_s):
            return None
        return self._TIMEOUT_TARGET.get(self.phase)

    @property
    def phase_label(self) -> str:
        return CORE_LABELS.get(self.phase, self.phase)

    # -------------------------------------------------------------- update
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
            self._enter(forced, timestamp_ms)
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
                    self._enter(candidate, timestamp_ms)
                self._pending_phase = None
                self._pending_count = 0
                self._pending_since_ms = None
        elif candidate == self.phase:
            self._pending_phase = None
            self._pending_count = 0
            self._pending_since_ms = None
        # When candidate is None the player is holding the current state.

        self._frames_in_phase += 1
        self._track_shot_metrics(features)
        return self.phase

    def _enter(self, state: str, timestamp_ms: Optional[int]) -> None:
        # The wrist peak is what "the hand has stopped going up" is measured
        # against, so it belongs to one attempt. Carrying it across attempts
        # let a previous shot's high point suppress the next shot's release.
        if state == "rise":
            self._wrist_peak = float("-inf")
            self._rise_floor = None
            self._climb = 0.0
            self._went_up = False
        self.phase = state
        self._phase_entered_ms = timestamp_ms
        self._frames_in_phase = 0
        self._pending_phase = None
        self._pending_count = 0

    def _confirmed(self, timestamp_ms: Optional[int]) -> bool:
        """Has the candidate state persisted long enough to be believed?"""
        if timestamp_ms is None or self._pending_since_ms is None:
            return self._pending_count >= self._hysteresis_frames
        held_s = (timestamp_ms - self._pending_since_ms) / 1000.0
        # A single frame can satisfy the duration at low frame rates; require
        # at least two observations so one noisy frame is never enough.
        return held_s >= self._hysteresis_s and self._pending_count >= 2

    def _dwelled(self, timestamp_ms: Optional[int]) -> bool:
        """Has the current state been held long enough to be allowed to exit?"""
        if self.phase == CORE_REST:
            return True
        if timestamp_ms is None or self._phase_entered_ms is None:
            return self._frames_in_phase >= self._min_dwell_frames
        return (timestamp_ms - self._phase_entered_ms) / 1000.0 >= self._min_dwell_s

    @staticmethod
    def _is_valid_transition(current: str, nxt: str) -> bool:
        allowed = CORE_TRANSITIONS.get(current, [])
        return nxt in allowed or nxt == CORE_REST

    def _track_shot_metrics(self, f: KinematicFeatures):
        level = self._wrist_level(f)
        if level is not None:
            self._track_climb(level)
        if f.knee_angle is not None and f.knee_angle < self._knee_min_angle:
            self._knee_min_angle = f.knee_angle
        if self.phase == CORE_REST:
            self._knee_min_angle = f.knee_angle or 180.0

    def _t(self, key: str, default: float) -> float:
        return float(self._thresholds.get(key, default))

    # ---------------------------------------------------------- wrist signal
    # The wrist drives most of the shot. Its world-space position is gated on
    # MediaPipe visibility and, when the gate rejects it, silently reads 0.0 --
    # indistinguishable from "the wrist is at hip height" in hip-centred
    # coordinates. On footage filmed from behind that happened on every frame
    # of a clip, so the machine watched a motionless wrist through an entire
    # jump shot. These helpers prefer world space when it was genuinely
    # observed and fall back to the image-space ratio, which survives the same
    # gate.

    @staticmethod
    def _wrist_level(f: KinematicFeatures) -> Optional[float]:
        """How high the shooting hand is, in whichever space is trustworthy."""
        if f.wrist_height_ratio is not None:
            return f.wrist_height_ratio
        if f.wrist_world_valid:
            return f.wrist_y
        return None

    # Where the two spaces disagree, the one reporting MOVEMENT is believed.
    #
    # They disagree only when one of them is blind, never when the hand
    # genuinely stayed still: a hand that did not move produces a small number
    # in both spaces. And blindness is asymmetric -- a rejected landmark reads
    # as no motion, never as motion that is not there.
    #
    # Preferring world whenever `wrist_world_valid` was set is not enough,
    # because the flag says the landmark passed the visibility gate, not that
    # it is any good. Measured on a jump shot filmed side-on: the world wrist
    # moved 0.065 m through a shot whose image-space signal moved 0.650 body
    # heights. The world reading was "valid" and useless, the machine sat in
    # `rise` for the whole clip, and an unmistakable 0.43-body-height jump was
    # thrown away.

    def _wrist_rising(self, f: KinematicFeatures) -> bool:
        by_world = f.wrist_world_valid and f.wrist_velocity_y > self._t(
            "ball_lift_wrist_velocity", 0.03
        )
        by_image = f.wrist_height_velocity > self._t(
            "ball_lift_wrist_rise_rate", 0.25
        )
        return by_world or by_image

    def _wrist_falling(self, f: KinematicFeatures) -> bool:
        by_world = f.wrist_world_valid and f.wrist_velocity_y < self._t(
            "follow_through_wrist_down_velocity", -0.02
        )
        by_image = f.wrist_height_velocity < -self._t(
            "ball_lift_wrist_rise_rate", 0.25
        )
        return by_world or by_image

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
        by_world = f.wrist_world_valid and f.wrist_y > f.hip_y_avg
        by_image = f.wrist_height_ratio is not None and f.wrist_height_ratio > 0.0
        return by_world or by_image

    def _track_climb(self, level: float) -> None:
        """Follow the hand's CURRENT upward run, not its highest point ever.

        The distinction is the whole point. Measuring from wherever the rise
        happened to be entered is not robust, because the rise is often entered
        on a blip while the ball is already carried high: on a ten-shot video
        the state was entered with the hand at 0.564 body heights, which was
        also the highest it went in that instant, so "has it climbed?" answered
        no for the following 27 seconds and every one of the ten shots was
        lost.

        Tracking a running minimum instead makes the answer self-correcting. A
        hand that comes back down sets a new floor, and the climb is measured
        from there -- which is exactly the motion a shot is: down into the
        carry, then up and away.
        """
        if self._rise_floor is None or level < self._rise_floor:
            self._rise_floor = level
            self._wrist_peak = level
        elif level > self._wrist_peak:
            self._wrist_peak = level
        self._climb = level - self._rise_floor

    @staticmethod
    def _wrist_at_shoulder(f: KinematicFeatures) -> bool:
        """Is the shooting hand at or above shoulder height?"""
        by_image = (
            f.wrist_above_shoulder_ratio is not None
            and f.wrist_above_shoulder_ratio >= 0.0
        )
        by_world = f.wrist_world_valid and f.wrist_y >= f.shoulder_y
        return by_image or by_world

    def _ball_went_up(self, f: KinematicFeatures) -> bool:
        """Is the ball in a position it can be released from?

        A shot is not "the arm happens to be extended"; it is the ball being
        carried up and let go. Two ways to establish that, and both are needed.

        Watching the carry happen is the primary one -- body heights when the
        image-space signal is available and metres otherwise, the two being
        close enough in magnitude for one threshold, the same assumption
        `_PROGRESS_EPSILON` already makes.

        But the carry is not always on camera. A clip cut close around one
        attempt can open with the hands already overhead, and then the climb
        is unobservable through no fault of the shot: two clips of an
        unmistakable jump shot -- 0.43 body heights of elevation -- were lost
        this way. A hand at shoulder height holding a ball IS the release
        position, however it got there, so seeing it is evidence in its own
        right rather than a relaxation of the rule.

        THE ANSWER LATCHES, and that is not a convenience. The question is
        about the past -- did the ball go up during this rise -- so it cannot
        become false again while the same rise is still running. Recomputing it
        per frame made it flicker, and flicker is fatal here: a release has to
        survive the confirmation window to be believed, and the evidence for it
        kept switching off between frames. Measured on a jump shot where the
        shooting wrist tracks poorly, the release condition fired on single
        frames three separate times and was never confirmed, so a jump of 0.43
        body heights produced no shot at all.
        """
        if not self._went_up:
            self._went_up = (
                self._climb >= self._t("release_min_wrist_climb", 0.10)
                or self._wrist_at_shoulder(f)
            )
        return self._went_up

    def _wrist_near_peak(self, f: KinematicFeatures) -> bool:
        level = self._wrist_level(f)
        if level is None or self._wrist_peak == float("-inf"):
            return False
        return level >= self._wrist_peak - self._t("release_peak_tolerance", 0.02)

    # ------------------------------------------------------------ transitions
    def _evaluate_transition(self, f: KinematicFeatures) -> Optional[str]:
        t = self._t

        if self.phase == CORE_REST:
            # "Is the ball in a position it could be shot from, and moving up?"
            #
            # The knee dip is deliberately NOT a way in any more. It used to
            # open the shot, which meant every re-settle and bounce on the toes
            # started an attempt. It is still analysed -- the tracker back-fills
            # frames from before the rise so the dip is always in the captured
            # window -- but it no longer decides anything.
            if not self._wrist_in_carry_window(f):
                return None
            if self._wrist_rising(f):
                return "rise"
            return None

        if self.phase == "rise":
            # Every way the old machine could reach `release`, from either the
            # `ball_lift` state or the `jump` state, gathered into one place.
            # Merging them is a strict gain: a shot that would have satisfied
            # the jump-side test while the machine happened to be sitting in
            # ball_lift used to be missed, because the test was in the other
            # state's branch.
            #
            # But merging them also removed a guard that used to be implicit.
            # Reaching `jump` previously meant the ball had already been
            # carried up through `ball_lift`, so a release condition could not
            # fire on a hand that had not moved. In one merged state nothing
            # enforced that, and the opening frames of a clip -- where the
            # One Euro filter, the ankle baseline and every velocity are still
            # settling from nothing -- produced a release inside 0.3 s.
            # Measured: it cost three separate fixtures. The false attempt was
            # correctly rejected each time, but the machine was then stuck in
            # `recovery` through the real shot that followed.
            #
            # `_ball_went_up` is the guard, and it says what a release means:
            # the ball cannot leave the hand on its way up until it has been
            # on its way up.
            if not self._ball_went_up(f) or not self._wrist_above_hips(f):
                return None

            elbow = f.elbow_angle
            elbow_extended = elbow is not None and elbow > t("release_elbow_min", 150)
            index_extended = (
                f.index_align_angle is not None
                and f.index_align_angle > t("release_index_align_min", 155)
            )

            # The hand has stopped going up and started coming down, or has
            # reached its high point and stalled there.
            #
            # REJECTED, on measurement: also requiring the elbow to have
            # extended during the rise. The reasoning was sound -- these two
            # tests describe CATCHING as well as they describe shooting, and a
            # free-throw clip was measured detecting four attempts where the
            # player took two, one extra being the player reaching up for the
            # returning ball. But reaching up for a ball extends the arm just
            # as straightening into a shot does, so the gate did not separate
            # them: the false attempt survived and a real shot in the ten-shot
            # video was lost (9 detected -> 8).
            #
            # Catching is not separable from shooting by pose alone. Both are
            # hands up, arm extended, hands down. The signal that distinguishes
            # them is the BALL -- moving away from the hands or towards them --
            # which is exactly what the ball tracker adds.
            if self._wrist_falling(f) or (
                self._wrist_near_peak(f) and not self._wrist_rising(f)
            ):
                return CORE_ANCHOR
            # The arm has straightened -- with the hand no longer driving up,
            # or with the finger snapping through.
            if elbow_extended and (
                not self._wrist_rising(f)
                or f.index_velocity_y > t("release_index_up_velocity", 0.04)
                or index_extended
            ):
                return CORE_ANCHOR
            # A set shot fires with a shallower arm than a jump shot.
            if (
                elbow is not None
                and elbow > t("set_shot_elbow_min", 145)
                and not self._wrist_rising(f)
            ):
                return CORE_ANCHOR
            # The finger is aligned and driving through the ball.
            if (
                f.index_align_angle is not None
                and f.index_align_angle > t("release_index_align_min", 150)
                and f.index_velocity_y > t("release_index_up_velocity", 0.03)
            ):
                return CORE_ANCHOR
            return None

        if self.phase == CORE_ANCHOR:
            index_follow = (
                f.index_align_angle is not None
                and f.index_align_angle > t("follow_through_index_align_min", 160)
            )
            # `self._wrist_falling(f)`, not the raw comparison it used to
            # inline. The helper is defined a few lines above, is used at the
            # other call site in this same method, and carries the
            # `wrist_world_valid` guard that the inline version dropped.
            #
            # Without it: `wrist_velocity_y` is computed as
            # `(wrist_y - prev_wrist) / dt`, and `wrist_y` is fabricated as 0.0
            # when the gate rejects the landmark. One rejected frame directly
            # after a valid high reading therefore produces a large negative
            # "velocity" out of nothing, which satisfies this condition
            # immediately and ends the release on a frame where the wrist was
            # never observed to fall -- truncating the follow-through window
            # that `follow_through_elbow` and `hold_duration_s` are measured in.
            if self._wrist_falling(f):
                return "recovery"
            if f.elbow_angle is not None and f.elbow_angle > t(
                "follow_through_elbow_min", 155
            ):
                return "recovery"
            if index_follow and f.index_velocity_y < t(
                "follow_through_index_down_velocity", -0.02
            ):
                return "recovery"
            return None

        if self.phase == "recovery":
            # Back on the floor and settled, OR the hand has come back down.
            # Either is enough: a set shot never leaves the floor, so waiting
            # for the feet to "return" to a baseline they never left is not a
            # signal, and a player who walks straight off after the shot never
            # settles at all.
            feet_down = (
                abs(f.ankle_y_avg - f.ankle_baseline_y)
                < t("landing_ankle_tolerance", 0.04)
            )
            settled = f.total_velocity < t("ready_max_velocity", 0.2)
            hand_down = not self._wrist_above_hips(f)
            if (feet_down and settled) or hand_down:
                return CORE_REST
            return None

        return None
