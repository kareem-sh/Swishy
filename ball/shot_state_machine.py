"""Event-driven basketball flight and made/missed state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Deque, Optional, Tuple

import numpy as np

from ball.models import BallDetection, BallSnapshot, RimDetection, ShotOutcome
from utils.config_loader import load_yaml


class BallShotState(str, Enum):
    """Chronological states for one physical shot attempt."""

    WAITING = "waiting"
    IN_HAND = "in_hand"
    RELEASED = "released"
    ASCENDING = "ascending"
    DESCENDING = "descending"
    RIM_APPROACH = "rim_approach"
    RIM_CONTACT = "rim_contact"
    CROSSED_INSIDE = "crossed_inside"
    CROSSED_OUTSIDE = "crossed_outside"
    MADE = "made"
    MISSED = "missed"
    UNKNOWN = "unknown"


class BallTrackingStatus(str, Enum):
    """Whether the current ball point is measured, predicted, or absent."""

    OBSERVED = "observed"
    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"


_TERMINAL_STATES = {
    BallShotState.MADE,
    BallShotState.MISSED,
    BallShotState.UNKNOWN,
}
_SHOT_START_PHASES = {"rise"}
_RELEASE_PHASES = {"release", "recovery"}
_FLIGHT_STATES = {
    BallShotState.RELEASED,
    BallShotState.ASCENDING,
    BallShotState.DESCENDING,
    BallShotState.RIM_APPROACH,
}
_ACTIVE_OUTCOME_STATES = _FLIGHT_STATES | {
    BallShotState.RIM_CONTACT,
    BallShotState.CROSSED_INSIDE,
    BallShotState.CROSSED_OUTSIDE,
}


@dataclass
class BallStateUpdate:
    """Observable state returned by :meth:`BallShotStateMachine.update`."""

    state: BallShotState
    tracking_status: BallTrackingStatus
    measurement_source: str = "none"
    outcome: Optional[ShotOutcome] = None
    released_this_frame: bool = False
    crossed_rim_this_frame: bool = False
    rim_center_xy: Optional[Tuple[float, float]] = None
    rim_inner_radius: Optional[float] = None
    crossing_xy: Optional[Tuple[float, float]] = None
    crossing_timestamp_ms: Optional[int] = None


class BallShotStateMachine:
    """Turn ball/rim observations into release, flight, and outcome events.

    Image coordinates are used throughout: x grows to the right and y grows
    downward.  A negative vertical velocity is therefore upward flight, while
    a positive vertical velocity is downward flight.
    """

    def __init__(self, config_name: str = "ball.yaml"):
        cfg = load_yaml(config_name)
        state_cfg = cfg.get("ball_state_machine", {})
        outcome_cfg = cfg.get("outcome", {})

        self.enabled = bool(state_cfg.get("enabled", True))
        self.dynamic_rim = bool(state_cfg.get("dynamic_rim", False))
        self.debug_outcome = bool(state_cfg.get("debug_outcome", False))
        self.release_distance_px = float(
            state_cfg.get("release_distance_px", 60)
        )
        self.release_distance_growth_px_s = float(
            state_cfg.get("release_distance_growth_px_s", 360.0)
        )
        self.release_min_speed_px_s = float(
            state_cfg.get("release_min_speed_px_s", 120)
        )
        self.ankle_release_height_ratio = float(
            state_cfg.get("ankle_release_height_ratio", 0.45)
        )
        self.ankle_release_fallback_px = float(
            state_cfg.get("ankle_release_fallback_px", 350.0)
        )
        self.minimum_player_height_px = float(
            state_cfg.get("minimum_player_height_px", 50.0)
        )
        self.ascending_velocity_px_s = float(
            state_cfg.get("ascending_velocity_px_s", 30)
        )
        self.descending_velocity_px_s = float(
            state_cfg.get("descending_velocity_px_s", 30)
        )
        self.approach_scale = float(
            state_cfg.get("rim_approach_radius_scale", 3.0)
        )
        self.above_margin_scale = float(
            state_cfg.get("rim_above_margin_scale", 0.25)
        )
        self.below_margin_scale = float(
            state_cfg.get("rim_below_margin_scale", 1.0)
        )
        self.net_channel_scale = float(
            state_cfg.get("net_channel_scale", 1.2)
        )
        self.rim_contact_channel_scale = float(
            state_cfg.get("rim_contact_channel_scale", 1.15)
        )
        self.rim_depth_scale = float(
            state_cfg.get("rim_depth_scale", 0.20)
        )
        self.rim_exit_channel_scale = float(
            state_cfg.get("rim_exit_channel_scale", 1.5)
        )
        self.entry_containment_tolerance_scale = max(
            0.0,
            float(state_cfg.get("entry_containment_tolerance_scale", 0.05)),
        )
        self.made_confirmation_s = max(
            0.0, float(state_cfg.get("made_confirmation_s", 0.0))
        )
        self.miss_confirmation_s = max(
            0.0, float(state_cfg.get("miss_confirmation_s", 0.05))
        )
        self.outcome_timeout_ms = max(
            1, int(state_cfg.get("outcome_timeout_ms", 3000))
        )
        self.lost_timeout_ms = max(
            1, int(state_cfg.get("lost_timeout_ms", 800))
        )
        self.rim_inner_scale = float(outcome_cfg.get("rim_inner_scale", 0.72))

        self._rim_samples: Deque[Tuple[float, float, float]] = deque(maxlen=10)
        self._previous_pose_phase: Optional[str] = None
        self.reset()

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def reset(self) -> None:
        """Clear session state, including the stabilized rim geometry."""
        self._rim_samples.clear()
        self.rim_center: Optional[Tuple[float, float]] = None
        self.rim_inner_radius: Optional[float] = None
        self._previous_pose_phase = None
        self.begin_attempt()

    def begin_attempt(self) -> None:
        """Clear per-shot evidence while preserving the static rim lock."""
        self.state = BallShotState.WAITING
        self.outcome: Optional[ShotOutcome] = None
        self.release_timestamp_ms: Optional[int] = None
        self.release_frame: Optional[int] = None
        self.entry_frame: Optional[int] = None
        self.entry_timestamp_ms: Optional[int] = None
        self.previous_snapshot: Optional[BallSnapshot] = None
        self.previous_observed_snapshot: Optional[BallSnapshot] = None
        self.previous_rim_center: Optional[Tuple[float, float]] = None
        self.previous_observed_rim_center: Optional[Tuple[float, float]] = None
        self.previous_wrist_distance: Optional[float] = None
        self.previous_wrist_timestamp_ms: Optional[int] = None
        self.last_observed_timestamp_ms: Optional[int] = None
        self.last_ball_radius: Optional[float] = None
        self.saw_ball_above_rim = False
        self.saw_inside_crossing = False
        self.saw_contact_center_inside_opening = False
        self.below_confirmation_count = 0
        self.miss_confirmation_count = 0
        self.below_confirmation_started_ms: Optional[int] = None
        self.miss_confirmation_started_ms: Optional[int] = None
        self.evidence: list[str] = []
        self.crossing_xy: Optional[Tuple[float, float]] = None
        self._rim_locked = False
        self._recorded_rim_contact = False
        self.saw_rim_contact = False

    def update(
        self,
        ball_detection: Optional[BallDetection],
        ball_snapshot: Optional[BallSnapshot],
        rim_detection: Optional[RimDetection],
        wrist_xy: Optional[Tuple[float, float]],
        ankle_y: Optional[float],
        pose_phase: Optional[str],
        timestamp_ms: int,
        player_height_px: Optional[float] = None,
    ) -> BallStateUpdate:
        """Process one frame and return the current physical shot state."""
        released_this_frame = False
        crossed_this_frame = False

        # A terminal result remains stable until the body begins another shot.
        if (
            self.terminal
            and pose_phase in _SHOT_START_PHASES
            and self._previous_pose_phase in (None, "ready")
        ):
            self.begin_attempt()

        if self.dynamic_rim or not self._rim_locked:
            self._update_rim(rim_detection)

        tracking_status = self._tracking_status(ball_detection, ball_snapshot)
        observed = tracking_status in {
            BallTrackingStatus.OBSERVED,
            BallTrackingStatus.TRACKED,
        }
        measurement_source = (
            ball_snapshot.measurement_source
            if ball_snapshot is not None
            else "none"
        )

        if observed and ball_detection is not None:
            self.last_ball_radius = float(ball_detection.radius)
            self.last_observed_timestamp_ms = timestamp_ms

        if not self.enabled or self.terminal:
            self._remember_pose_phase(pose_phase)
            return self._result(
                tracking_status, measurement_source=measurement_source
            )

        if ball_snapshot is None:
            self._check_timeouts(timestamp_ms, tracking_status)
            self._remember_pose_phase(pose_phase)
            return self._result(
                tracking_status, measurement_source=measurement_source
            )

        wrist_distance = self._ball_wrist_distance(ball_snapshot, wrist_xy)
        vx, vy = ball_snapshot.velocity_xy
        speed = math.hypot(vx, vy)
        # print(
        #     ball_snapshot.frame_index,
        #     ball_snapshot.x,
        #     ball_snapshot.y,
        #     ball_snapshot.velocity_xy,
        #     ball_snapshot.is_interpolated,
        #     speed,
        # )
        relative_vy = self._relative_vertical_velocity(ball_snapshot, vy)

        if self.state == BallShotState.WAITING and wrist_distance is not None:
            in_hand_limit = self.release_distance_px
            if pose_phase in {"rise"}:
                in_hand_limit *= 1.5
            if wrist_distance <= in_hand_limit:
                self.state = BallShotState.IN_HAND

        if self.state in {BallShotState.WAITING, BallShotState.IN_HAND}:
            if self._release_confirmed(
                pose_phase=pose_phase,
                ball_snapshot=ball_snapshot,
                wrist_distance=wrist_distance,
                speed=speed,
                ankle_y=ankle_y,
                player_height_px=player_height_px,
                timestamp_ms=timestamp_ms,
            ):
                print("ball release confirmed")
                self.state = BallShotState.RELEASED
                self.release_timestamp_ms = timestamp_ms
                self.release_frame = ball_snapshot.frame_index
                self.evidence.append("Ball release confirmed")
                released_this_frame = True

        if self.state in _ACTIVE_OUTCOME_STATES:
            if self.state == BallShotState.RELEASED:
                if relative_vy < -self.ascending_velocity_px_s:
                    self.state = BallShotState.ASCENDING
                elif relative_vy > self.descending_velocity_px_s:
                    self.state = BallShotState.DESCENDING
                    if not self.dynamic_rim:
                        self._rim_locked = self.rim_center is not None
            elif self.state == BallShotState.ASCENDING:
                if relative_vy >= self.descending_velocity_px_s:
                    self.state = BallShotState.DESCENDING
                    if not self.dynamic_rim:
                        self._rim_locked = self.rim_center is not None
                    self.evidence.append("Ball passed trajectory apex")

            crossed_this_frame = self._process_rim_geometry(
                snapshot=ball_snapshot,
                observed=observed,
                vertical_velocity=relative_vy,
                timestamp_ms=timestamp_ms,
            )

        if self.outcome is None:
            self._check_timeouts(timestamp_ms, tracking_status)

        self.previous_snapshot = ball_snapshot
        self.previous_rim_center = self.rim_center
        if observed:
            self.previous_observed_snapshot = ball_snapshot
            self.previous_observed_rim_center = self.rim_center
        if wrist_distance is not None:
            self.previous_wrist_distance = wrist_distance
            self.previous_wrist_timestamp_ms = timestamp_ms
        self._remember_pose_phase(pose_phase)

        return self._result(
            tracking_status,
            measurement_source=measurement_source,
            released_this_frame=released_this_frame,
            crossed_rim_this_frame=crossed_this_frame,
        )

    def _tracking_status(
        self,
        detection: Optional[BallDetection],
        snapshot: Optional[BallSnapshot],
    ) -> BallTrackingStatus:
        if detection is not None and snapshot is not None:
            if detection.measurement_source == "nanotrack":
                return BallTrackingStatus.TRACKED
            if not detection.is_visual_observation:
                return BallTrackingStatus.PREDICTED
            return BallTrackingStatus.OBSERVED
        if snapshot is not None and snapshot.is_interpolated:
            return BallTrackingStatus.PREDICTED
        return BallTrackingStatus.LOST

    def _update_rim(self, rim: Optional[RimDetection]) -> None:
        if rim is None:
            return
        x1, _, x2, _ = rim.bbox_xyxy
        inner_radius = max(1.0, (x2 - x1) * self.rim_inner_scale / 2.0)
        if self.dynamic_rim:
            self.rim_center = (float(rim.x), float(rim.y))
            self.rim_inner_radius = inner_radius
            return
        self._rim_samples.append((float(rim.x), float(rim.y), inner_radius))
        samples = np.asarray(self._rim_samples, dtype=np.float64)
        self.rim_center = (
            float(np.median(samples[:, 0])),
            float(np.median(samples[:, 1])),
        )
        self.rim_inner_radius = float(np.median(samples[:, 2]))

    def _relative_vertical_velocity(
        self,
        snapshot: BallSnapshot,
        fallback_velocity_y: float,
    ) -> float:
        """Measure ball motion relative to the moving rim, not the image."""
        if (
            self.rim_center is None
            or self.previous_rim_center is None
            or self.previous_snapshot is None
        ):
            return fallback_velocity_y

        dt = (snapshot.timestamp_ms - self.previous_snapshot.timestamp_ms) / 1000.0
        if dt <= 1e-6:
            return fallback_velocity_y
        previous_relative_y = (
            self.previous_snapshot.y - self.previous_rim_center[1]
        )
        current_relative_y = snapshot.y - self.rim_center[1]
        return (current_relative_y - previous_relative_y) / dt

    def _release_confirmed(
        self,
        pose_phase: Optional[str],
        ball_snapshot: Optional[BallSnapshot],
        wrist_distance: Optional[float],
        speed: float,
        ankle_y: Optional[float],
        player_height_px: Optional[float],
        timestamp_ms: int,
    ) -> bool:
        pose_release = pose_phase in _RELEASE_PHASES
        difference_from_ankle = (
            ankle_y - ball_snapshot.y
            if ankle_y is not None and ball_snapshot is not None
            else None
        )
        valid_player_height = (
            player_height_px is not None
            and math.isfinite(player_height_px)
            and player_height_px >= self.minimum_player_height_px
        )
        if difference_from_ankle is None:
            ankle_release = False
        elif valid_player_height:
            # Body-height units make the same gate work across resolutions,
            # zoom levels, and different distances from the camera.
            ankle_release = (
                difference_from_ankle / player_height_px
                >= self.ankle_release_height_ratio
            )
        else:
            # Preserve usable behavior on frames where MediaPipe cannot provide
            # a trustworthy nose-to-ankle scale.
            ankle_release = (
                difference_from_ankle >= self.ankle_release_fallback_px
            )
        ball_fast = speed >= self.release_min_speed_px_s
        far_from_wrist = (
            wrist_distance is not None
            and wrist_distance >= self.release_distance_px * 1.5
        )
        separating = False
        if (
            wrist_distance is not None
            and self.previous_wrist_distance is not None
            and self.previous_wrist_timestamp_ms is not None
        ):
            dt_s = (timestamp_ms - self.previous_wrist_timestamp_ms) / 1000.0
            distance_growth_px_s = (
                (wrist_distance - self.previous_wrist_distance) / dt_s
                if dt_s > 1e-6
                else 0.0
            )
            separating = (
                wrist_distance >= self.release_distance_px
                and distance_growth_px_s >= self.release_distance_growth_px_s
            )
        # print(separating , ball_fast , ankle_release)
        return ankle_release and(
            (pose_release and separating and ball_fast and far_from_wrist)
        )

    @staticmethod
    def _ball_wrist_distance(
        snapshot: BallSnapshot,
        wrist_xy: Optional[Tuple[float, float]],
    ) -> Optional[float]:
        if wrist_xy is None:
            return None
        return math.hypot(snapshot.x - wrist_xy[0], snapshot.y - wrist_xy[1])

    def _process_rim_geometry(
        self,
        snapshot: BallSnapshot,
        observed: bool,
        vertical_velocity: float,
        timestamp_ms: int,
    ) -> bool:
        if self.rim_center is None or self.rim_inner_radius is None:
            return False

        rim_x, rim_y = self.rim_center
        radius = self._effective_ball_radius()
        horizontal_offset = abs(snapshot.x - rim_x)
        distance = math.hypot(snapshot.x - rim_x, snapshot.y - rim_y)
        near_rim = distance <= self.rim_inner_radius * self.approach_scale
        rim_depth = max(2.0, self.rim_inner_radius * self.rim_depth_scale)
        rim_top = rim_y - rim_depth
        rim_bottom = rim_y + rim_depth

        if near_rim and self.state in _FLIGHT_STATES:
            self.state = BallShotState.RIM_APPROACH
            if not self.dynamic_rim:
                self._rim_locked = True

        above_margin = radius * self.above_margin_scale
        newly_above_rim = (
            observed
            and near_rim
            and snapshot.y < rim_top - above_margin
            and not self.saw_ball_above_rim
        )
        if observed and near_rim and snapshot.y < rim_top - above_margin:
            self.saw_ball_above_rim = True
        if newly_above_rim:
            self._debug_outcome_event(
                "ABOVE_RIM",
                frame=snapshot.frame_index,
                ball_y=snapshot.y,
                rim_y=rim_y,
                rim_top=rim_top,
                vertical_velocity=vertical_velocity,
            )

        self._record_possible_rim_contact(snapshot, observed, radius)

        crossed = False
        previous = self.previous_snapshot
        if previous is not None and self.state == BallShotState.RIM_APPROACH:
            previous_rim = self.previous_rim_center or (rim_x, rim_y)
            previous_relative_x = previous.x - previous_rim[0]
            previous_relative_y = previous.y - previous_rim[1]
            current_relative_x = snapshot.x - rim_x
            current_relative_y = snapshot.y - rim_y
            downward_crossing = previous_relative_y < 0.0 <= current_relative_y
            # Do not declare the entry from a constant-velocity prediction.
            # Two measured endpoints may still straddle the rim plane; the
            # crossing point is interpolated between them below.
            crossing_supported = observed and not previous.is_interpolated
            if (
                downward_crossing
                and crossing_supported
                and self.saw_ball_above_rim
                and vertical_velocity >= 0
            ):
                crossing_relative_x = self._interpolate_relative_crossing_x(
                    previous_relative_x,
                    previous_relative_y,
                    current_relative_x,
                    current_relative_y,
                )
                if crossing_relative_x is not None:
                    crossing_x = rim_x + crossing_relative_x
                    crossed = True
                    self.crossing_xy = (crossing_x, rim_y)
                    self.entry_frame = snapshot.frame_index
                    crossing_ratio = self._interpolate_crossing_ratio(
                        previous_relative_y,
                        current_relative_y,
                    )
                    self.entry_timestamp_ms = (
                        int(
                            round(
                                previous.timestamp_ms
                                + crossing_ratio
                                * (snapshot.timestamp_ms - previous.timestamp_ms)
                            )
                        )
                        if crossing_ratio is not None
                        else snapshot.timestamp_ms
                    )
                    crossing_offset = abs(crossing_relative_x)
                    contact_limit = (
                        self.rim_inner_radius * self.rim_contact_channel_scale
                    )
                    contour_fits = self._ball_contour_fits_opening(
                        crossing_relative_x,
                        radius,
                    )
                    if contour_fits:
                        self.state = BallShotState.CROSSED_INSIDE
                        self.saw_inside_crossing = True
                        self.evidence.append(
                            "Ball contour fit inside rim opening at downward crossing"
                        )
                    elif crossing_offset <= contact_limit:
                        # An edge crossing near the physical opening can still
                        # deflect inward. Keep it unresolved until the tracked
                        # center is observed inside the actual inner opening.
                        self.state = BallShotState.RIM_CONTACT
                        self.saw_rim_contact = True
                        self.saw_contact_center_inside_opening = (
                            crossing_offset <= self.rim_inner_radius
                        )
                        self.evidence.append("Ball crossed rim contact zone")
                    else:
                        self.state = BallShotState.CROSSED_OUTSIDE
                        self.evidence.append("Ball crossed outside rim opening")

                    self._debug_outcome_event(
                        "RIM_CROSSING",
                        frame=snapshot.frame_index,
                        classification=self.state.value,
                        crossing_x=crossing_x,
                        crossing_offset=abs(crossing_relative_x),
                        ball_radius=radius,
                        rim_inner_radius=self.rim_inner_radius,
                        contour_fits=contour_fits,
                        saw_rim_contact=self.saw_rim_contact,
                        vertical_velocity=vertical_velocity,
                    )

        below_threshold = max(
            rim_y + radius * self.below_margin_scale,
            rim_bottom + radius * 0.25,
        )
        below_rim = snapshot.y > below_threshold
        inside_net_channel = (
            horizontal_offset <= self.rim_inner_radius * self.net_channel_scale
        )
        inside_inner_opening = horizontal_offset <= self.rim_inner_radius
        if (
            self.state == BallShotState.RIM_CONTACT
            and observed
            and inside_inner_opening
            and not below_rim
        ):
            # Preserve the entry evidence. A made ball can move diagonally
            # after crossing the opening, so it need not remain within the
            # inner radius on the later frame that confirms it is below.
            self.saw_contact_center_inside_opening = True
        previous_distance = None
        if (
            self.previous_observed_snapshot is not None
            and self.previous_observed_rim_center is not None
        ):
            previous_distance = math.hypot(
                self.previous_observed_snapshot.x
                - self.previous_observed_rim_center[0],
                self.previous_observed_snapshot.y
                - self.previous_observed_rim_center[1],
            )
        moving_away = (
            previous_distance is not None
            and distance
            > previous_distance + max(2.0, self.rim_inner_radius * 0.05)
        )
        rebounding_up = (
            observed
            and vertical_velocity < -self.ascending_velocity_px_s
            and snapshot.y < rim_bottom
        )
        outside_exit = (
            observed
            and horizontal_offset
            > self.rim_inner_radius * self.rim_exit_channel_scale
            and moving_away
        )

        if self.state in {
            BallShotState.CROSSED_INSIDE,
            BallShotState.RIM_CONTACT,
        }:
            # A clean contour-fit crossing already proves that the ball passed
            # through the rim opening. This must not depend on a vertical net
            # channel: without a net, the ball naturally continues diagonally.
            # Rim-contact entries remain stricter because they can still
            # deflect away after touching the rim.
            clean_inside_exit = (
                self.state == BallShotState.CROSSED_INSIDE
                and self.saw_inside_crossing
                and observed
                and below_rim
                and vertical_velocity >= 0.0
            )
            contact_inside_exit = (
                self.state == BallShotState.RIM_CONTACT
                and self.saw_contact_center_inside_opening
                and observed
                and below_rim
                and vertical_velocity >= 0.0
            )
            confirmed_made_exit = clean_inside_exit or contact_inside_exit
            unverified_contact_exit = (
                self.state == BallShotState.RIM_CONTACT
                and observed
                and below_rim
                and not self.saw_contact_center_inside_opening
            )
            miss_evidence = rebounding_up or (
                self.state == BallShotState.RIM_CONTACT
                and not self.saw_contact_center_inside_opening
                and (outside_exit or unverified_contact_exit)
            )
            make_duration_s = self._update_made_confirmation(
                confirmed_made_exit, observed, timestamp_ms
            )
            miss_duration_s = self._update_miss_confirmation(
                miss_evidence, observed, timestamp_ms
            )

            self._debug_outcome_event(
                "POST_ENTRY",
                frame=snapshot.frame_index,
                state=self.state.value,
                observed=observed,
                below_rim=below_rim,
                inside_net_channel=inside_net_channel,
                inside_inner_opening=inside_inner_opening,
                saw_contact_inner_entry=(
                    self.saw_contact_center_inside_opening
                ),
                confirmation_mode=(
                    "clean_crossing"
                    if self.state == BallShotState.CROSSED_INSIDE
                    else "rim_contact"
                ),
                rebounding_up=rebounding_up,
                outside_exit=outside_exit,
                make_duration_s=(
                    f"{make_duration_s:.3f}/{self.made_confirmation_s:.3f}"
                ),
                miss_duration_s=(
                    f"{miss_duration_s:.3f}/{self.miss_confirmation_s:.3f}"
                ),
            )

            if (
                confirmed_made_exit
                and make_duration_s >= self.made_confirmation_s
            ):
                contact_make = self.state == BallShotState.RIM_CONTACT
                self._finish(
                    result="made",
                    confidence=0.90 if contact_make else 0.95,
                    evidence=(
                        "Ball center crossed the inner opening and descended below the rim after rim contact"
                        if contact_make
                        else "Ball observed below rim after clean inside crossing"
                    ),
                    timestamp_ms=timestamp_ms,
                )
            elif miss_evidence and miss_duration_s >= self.miss_confirmation_s:
                if rebounding_up:
                    miss_reason = "Ball rebounded upward after rim crossing"
                elif outside_exit:
                    miss_reason = "Ball exited away from the basket after rim contact"
                else:
                    miss_reason = (
                        "Ball remained below the rim outside the inner opening"
                    )
                self._finish(
                    result="missed",
                    confidence=0.85,
                    evidence=miss_reason,
                    timestamp_ms=timestamp_ms,
                )

        elif self.state == BallShotState.CROSSED_OUTSIDE:
            miss_evidence = (
                observed
                and (below_rim or rebounding_up or outside_exit)
            )
            self._update_made_confirmation(False, observed, timestamp_ms)
            miss_duration_s = self._update_miss_confirmation(
                miss_evidence, observed, timestamp_ms
            )
            self._debug_outcome_event(
                "OUTSIDE_ENTRY",
                frame=snapshot.frame_index,
                below_rim=below_rim,
                inside_net_channel=inside_net_channel,
                inside_inner_opening=inside_inner_opening,
                rebounding_up=rebounding_up,
                outside_exit=outside_exit,
                miss_duration_s=(
                    f"{miss_duration_s:.3f}/{self.miss_confirmation_s:.3f}"
                ),
            )
            if miss_evidence and miss_duration_s >= self.miss_confirmation_s:
                self._finish(
                    result="missed",
                    confidence=0.90,
                    evidence="Ball passed below rim after crossing outside opening",
                    timestamp_ms=timestamp_ms,
                )

        elif self.state == BallShotState.RIM_APPROACH:
            # Handles an airball when sparse frames do not capture the exact
            # plane crossing but the ball is visibly below and well outside.
            miss_evidence = (
                observed
                and below_rim
                and horizontal_offset
                > self.rim_inner_radius * self.rim_exit_channel_scale
            )
            self._update_made_confirmation(False, observed, timestamp_ms)
            miss_duration_s = self._update_miss_confirmation(
                miss_evidence, observed, timestamp_ms
            )
            if observed and below_rim:
                self._debug_outcome_event(
                    "AIRBALL_CHECK",
                    frame=snapshot.frame_index,
                    horizontal_offset=horizontal_offset,
                    exit_limit=(
                        self.rim_inner_radius * self.rim_exit_channel_scale
                    ),
                    miss_duration_s=(
                        f"{miss_duration_s:.3f}/{self.miss_confirmation_s:.3f}"
                    ),
                )
            if miss_evidence and miss_duration_s >= self.miss_confirmation_s:
                self._finish(
                    result="missed",
                    confidence=0.80,
                    evidence="Ball continued below and outside the basket",
                    timestamp_ms=timestamp_ms,
                )

        return crossed

    def _update_made_confirmation(
        self, condition: bool, observed: bool, timestamp_ms: int
    ) -> float:
        if not observed or not condition:
            self.below_confirmation_started_ms = None
            self.below_confirmation_count = 0
            return 0.0
        if self.below_confirmation_started_ms is None:
            self.below_confirmation_started_ms = timestamp_ms
        self.below_confirmation_count += 1
        return max(
            0.0,
            (timestamp_ms - self.below_confirmation_started_ms) / 1000.0,
        )

    def _update_miss_confirmation(
        self, condition: bool, observed: bool, timestamp_ms: int
    ) -> float:
        if not observed or not condition:
            self.miss_confirmation_started_ms = None
            self.miss_confirmation_count = 0
            return 0.0
        if self.miss_confirmation_started_ms is None:
            self.miss_confirmation_started_ms = timestamp_ms
        self.miss_confirmation_count += 1
        return max(
            0.0,
            (timestamp_ms - self.miss_confirmation_started_ms) / 1000.0,
        )

    def _ball_contour_fits_opening(
        self,
        crossing_relative_x: float,
        ball_radius: float,
    ) -> bool:
        """Whether the ball's horizontal circle fits in the inner rim opening.

        The detector provides boxes rather than segmentation masks, so the
        ball contour is represented by its tracked center and effective
        radius. At the rim-entry plane, horizontal containment is the useful
        2D test; vertical containment would reject a ball precisely while it
        is passing through that plane.
        """
        if self.rim_inner_radius is None:
            return False

        tolerance = (
            self.rim_inner_radius * self.entry_containment_tolerance_scale
        )
        opening_left = -self.rim_inner_radius - tolerance
        opening_right = self.rim_inner_radius + tolerance
        ball_left = crossing_relative_x - ball_radius
        ball_right = crossing_relative_x + ball_radius
        return ball_left >= opening_left and ball_right <= opening_right

    def _debug_outcome_event(self, event: str, **values) -> None:
        """Print concise evidence explaining a made/missed state decision."""
        if not self.debug_outcome:
            return

        parts = []
        for key, value in values.items():
            if isinstance(value, float):
                rendered = f"{value:.2f}"
            else:
                rendered = str(value)
            parts.append(f"{key}={rendered}")
        details = " ".join(parts)
        print(f"[BALL OUTCOME] {event} {details}".rstrip())

    def _effective_ball_radius(self) -> float:
        if self.rim_inner_radius is None:
            return max(1.0, self.last_ball_radius or 1.0)
        radius = self.last_ball_radius or self.rim_inner_radius * 0.45
        return min(
            max(float(radius), self.rim_inner_radius * 0.15),
            self.rim_inner_radius * 0.75,
        )

    def _record_possible_rim_contact(
        self,
        snapshot: BallSnapshot,
        observed: bool,
        ball_radius: float,
    ) -> None:
        if (
            self._recorded_rim_contact
            or not observed
            or self.rim_center is None
            or self.rim_inner_radius is None
        ):
            return
        rim_x, rim_y = self.rim_center
        thickness = max(2.0, self.rim_inner_radius * 0.08)
        left_distance = math.hypot(
            snapshot.x - (rim_x - self.rim_inner_radius), snapshot.y - rim_y
        )
        right_distance = math.hypot(
            snapshot.x - (rim_x + self.rim_inner_radius), snapshot.y - rim_y
        )
        if min(left_distance, right_distance) <= ball_radius + thickness:
            # A single-camera 2D overlap can also mean that the ball passed in
            # front of or behind the rim. Record it as diagnostic evidence,
            # but do not promote it to verified physical rim contact.
            self.evidence.append("Possible projected rim-edge overlap")
            self._recorded_rim_contact = True

    @staticmethod
    def _interpolate_relative_crossing_x(
        previous_x: float,
        previous_y: float,
        current_x: float,
        current_y: float,
    ) -> Optional[float]:
        """Interpolate horizontal offset where relative vertical offset is zero."""
        dy = current_y - previous_y
        if abs(dy) < 1e-6:
            return None
        ratio = -previous_y / dy
        if not 0.0 <= ratio <= 1.0:
            return None
        return previous_x + ratio * (current_x - previous_x)

    @staticmethod
    def _interpolate_crossing_ratio(
        previous_y: float,
        current_y: float,
    ) -> Optional[float]:
        """Return where in a frame interval the rim plane was crossed."""
        dy = current_y - previous_y
        if abs(dy) < 1e-6:
            return None
        ratio = -previous_y / dy
        return ratio if 0.0 <= ratio <= 1.0 else None

    def _check_timeouts(
        self,
        timestamp_ms: int,
        tracking_status: BallTrackingStatus,
    ) -> None:
        if self.release_timestamp_ms is None or self.outcome is not None:
            return

        if (
            self.state
            in {
                BallShotState.RIM_APPROACH,
                BallShotState.RIM_CONTACT,
                BallShotState.CROSSED_INSIDE,
                BallShotState.CROSSED_OUTSIDE,
            }
            and self.last_observed_timestamp_ms is not None
            and timestamp_ms - self.last_observed_timestamp_ms
            >= self.lost_timeout_ms
        ):
            self._finish(
                result="unknown",
                confidence=0.20,
                evidence="Ball lost near rim",
                timestamp_ms=timestamp_ms,
            )
            return

        if timestamp_ms - self.release_timestamp_ms < self.outcome_timeout_ms:
            return

        if self.state in {
            BallShotState.RIM_CONTACT,
            BallShotState.CROSSED_INSIDE,
        }:
            self._finish(
                result="unknown",
                confidence=0.30,
                evidence="Rim interaction remained ambiguous before timeout",
                timestamp_ms=timestamp_ms,
            )
        elif tracking_status == BallTrackingStatus.OBSERVED:
            self._finish(
                result="missed",
                confidence=0.65,
                evidence="No valid inside rim crossing before timeout",
                timestamp_ms=timestamp_ms,
            )
        else:
            self._finish(
                result="unknown",
                confidence=0.20,
                evidence="Outcome timeout with incomplete ball tracking",
                timestamp_ms=timestamp_ms,
            )

    def _finish(
        self,
        result: str,
        confidence: float,
        evidence: str,
        timestamp_ms: int,
    ) -> ShotOutcome:
        if self.outcome is not None:
            return self.outcome

        self.evidence.append(evidence)
        self._debug_outcome_event(
            "FINAL",
            result=result.upper(),
            confidence=confidence,
            reason=evidence,
            saw_ball_above_rim=self.saw_ball_above_rim,
            saw_inside_crossing=self.saw_inside_crossing,
            saw_rim_contact=self.saw_rim_contact,
            make_count=self.below_confirmation_count,
            miss_count=self.miss_confirmation_count,
        )
        self.state = {
            "made": BallShotState.MADE,
            "missed": BallShotState.MISSED,
            "unknown": BallShotState.UNKNOWN,
        }[result]
        self.outcome = ShotOutcome(
            result=result,
            confidence=confidence,
            release_frame=self.release_frame,
            release_timestamp_ms=self.release_timestamp_ms,
            entry_frame=self.entry_frame,
            entry_timestamp_ms=self.entry_timestamp_ms,
            outcome_timestamp_ms=timestamp_ms,
            evidence=list(self.evidence),
            timeseries_summary={
                "ball_state": self.state.value,
                "saw_ball_above_rim": self.saw_ball_above_rim,
                "saw_inside_crossing": self.saw_inside_crossing,
                "saw_rim_contact": self.saw_rim_contact,
                "saw_contact_center_inside_opening": (
                    self.saw_contact_center_inside_opening
                ),
                "crossing_xy": self.crossing_xy,
                "crossing_timestamp_ms": self.entry_timestamp_ms,
                "below_confirmation_count": self.below_confirmation_count,
                "miss_confirmation_count": self.miss_confirmation_count,
                "made_confirmation_s": self.made_confirmation_s,
                "miss_confirmation_s": self.miss_confirmation_s,
            },
        )
        return self.outcome

    def _result(
        self,
        tracking_status: BallTrackingStatus,
        measurement_source: str = "none",
        released_this_frame: bool = False,
        crossed_rim_this_frame: bool = False,
    ) -> BallStateUpdate:
        return BallStateUpdate(
            state=self.state,
            tracking_status=tracking_status,
            measurement_source=measurement_source,
            outcome=self.outcome,
            released_this_frame=released_this_frame,
            crossed_rim_this_frame=crossed_rim_this_frame,
            rim_center_xy=self.rim_center,
            rim_inner_radius=self.rim_inner_radius,
            crossing_xy=self.crossing_xy,
            crossing_timestamp_ms=self.entry_timestamp_ms,
        )

    def _remember_pose_phase(self, pose_phase: Optional[str]) -> None:
        if pose_phase is not None:
            self._previous_pose_phase = pose_phase
