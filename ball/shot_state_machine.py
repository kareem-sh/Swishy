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
    PREDICTED = "predicted"
    LOST = "lost"


_TERMINAL_STATES = {
    BallShotState.MADE,
    BallShotState.MISSED,
    BallShotState.UNKNOWN,
}
_SHOT_START_PHASES = {"loading", "ball_lift"}
_RELEASE_PHASES = {"release", "follow_through"}
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
    outcome: Optional[ShotOutcome] = None
    released_this_frame: bool = False
    crossed_rim_this_frame: bool = False
    rim_center_xy: Optional[Tuple[float, float]] = None
    rim_inner_radius: Optional[float] = None
    crossing_xy: Optional[Tuple[float, float]] = None


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
        self.release_distance_px = float(
            state_cfg.get("release_distance_px", 60)
        )
        self.release_distance_growth_px = float(
            state_cfg.get("release_distance_growth_px", 12)
        )
        self.release_min_speed_px_s = float(
            state_cfg.get("release_min_speed_px_s", 120)
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
        self.below_confirmation_frames = max(
            1, int(state_cfg.get("below_confirmation_frames", 2))
        )
        self.miss_confirmation_frames = max(
            1, int(state_cfg.get("miss_confirmation_frames", 2))
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
        self.previous_snapshot: Optional[BallSnapshot] = None
        self.previous_observed_snapshot: Optional[BallSnapshot] = None
        self.previous_rim_center: Optional[Tuple[float, float]] = None
        self.previous_observed_rim_center: Optional[Tuple[float, float]] = None
        self.previous_wrist_distance: Optional[float] = None
        self.last_observed_timestamp_ms: Optional[int] = None
        self.last_ball_radius: Optional[float] = None
        self.saw_ball_above_rim = False
        self.saw_inside_crossing = False
        self.below_confirmation_count = 0
        self.miss_confirmation_count = 0
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
        ankle_y: float,
        pose_phase: Optional[str],
        timestamp_ms: int,
    ) -> BallStateUpdate:
        """Process one frame and return the current physical shot state."""
        released_this_frame = False
        crossed_this_frame = False

        # A terminal result remains stable until the body begins another shot.
        if (
            self.terminal
            and pose_phase in _SHOT_START_PHASES
            and self._previous_pose_phase in (None, "ready_stance")
        ):
            self.begin_attempt()

        if self.dynamic_rim or not self._rim_locked:
            self._update_rim(rim_detection)

        tracking_status = self._tracking_status(ball_detection, ball_snapshot)
        observed = tracking_status == BallTrackingStatus.OBSERVED

        if observed and ball_detection is not None:
            self.last_ball_radius = float(ball_detection.radius)
            self.last_observed_timestamp_ms = timestamp_ms

        if not self.enabled or self.terminal:
            self._remember_pose_phase(pose_phase)
            return self._result(tracking_status)

        if ball_snapshot is None:
            self._check_timeouts(timestamp_ms, tracking_status)
            self._remember_pose_phase(pose_phase)
            return self._result(tracking_status)

        wrist_distance = self._ball_wrist_distance(ball_snapshot, wrist_xy)
        vx, vy = ball_snapshot.velocity_xy
        speed = math.hypot(vx, vy)
        relative_vy = self._relative_vertical_velocity(ball_snapshot, vy)

        if self.state == BallShotState.WAITING and wrist_distance is not None:
            in_hand_limit = self.release_distance_px
            if pose_phase in {"loading", "knee_flexion", "ball_lift"}:
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
            ):
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
        self._remember_pose_phase(pose_phase)

        return self._result(
            tracking_status,
            released_this_frame=released_this_frame,
            crossed_rim_this_frame=crossed_this_frame,
        )

    def _tracking_status(
        self,
        detection: Optional[BallDetection],
        snapshot: Optional[BallSnapshot],
    ) -> BallTrackingStatus:
        if detection is not None and snapshot is not None:
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
        ankle_y: float,
    ) -> bool:
        pose_release = pose_phase in _RELEASE_PHASES
        difference_from_ankle = (
            ankle_y - ball_snapshot.y
            if ankle_y is not None and ball_snapshot is not None
            else None
        )
        print(difference_from_ankle)
        ankle_threshold = 350.0  # pixels, adjust as needed
        ankle_release = (
            difference_from_ankle is not None
            and difference_from_ankle > ankle_threshold
        )
        ball_fast = speed >= self.release_min_speed_px_s
        far_from_wrist = (
            wrist_distance is not None
            and wrist_distance >= self.release_distance_px * 1.5
        )
        separating = False
        if wrist_distance is not None and self.previous_wrist_distance is not None:
            separating = (
                wrist_distance >= self.release_distance_px
                and wrist_distance - self.previous_wrist_distance
                >= self.release_distance_growth_px
                
            )

        return (
            pose_release and ((separating or ball_fast or far_from_wrist) and ankle_release)
        ) 
    # or (separating and ball_fast  and ankle_release)

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
        if observed and near_rim and snapshot.y < rim_top - above_margin:
            self.saw_ball_above_rim = True

        self._record_possible_rim_contact(snapshot, observed, radius)

        in_contact_zone = (
            horizontal_offset
            <= self.rim_inner_radius * self.rim_contact_channel_scale
            and rim_top - radius <= snapshot.y <= rim_bottom + radius
        )
        if (
            self.state == BallShotState.CROSSED_OUTSIDE
            and observed
            and in_contact_zone
        ):
            self.state = BallShotState.RIM_CONTACT
            self.saw_rim_contact = True
            self.evidence.append("Ball re-entered rim contact zone")

        crossed = False
        previous = self.previous_snapshot
        if previous is not None and self.state == BallShotState.RIM_APPROACH:
            previous_rim = self.previous_rim_center or (rim_x, rim_y)
            previous_relative_x = previous.x - previous_rim[0]
            previous_relative_y = previous.y - previous_rim[1]
            current_relative_x = snapshot.x - rim_x
            current_relative_y = snapshot.y - rim_y
            downward_crossing = previous_relative_y < 0.0 <= current_relative_y
            crossing_supported = observed or not previous.is_interpolated
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
                    clearance = max(
                        self.rim_inner_radius - radius,
                        self.rim_inner_radius * 0.20,
                    )
                    crossing_offset = abs(crossing_relative_x)
                    contact_limit = (
                        self.rim_inner_radius * self.rim_contact_channel_scale
                    )
                    if crossing_offset <= clearance:
                        self.state = BallShotState.CROSSED_INSIDE
                        self.saw_inside_crossing = True
                        self.evidence.append("Ball crossed inside rim opening")
                    elif crossing_offset <= contact_limit or self.saw_rim_contact:
                        # Edge/front/back rim makes can cross outside the strict
                        # center clearance and then deflect into the net. Keep
                        # this result unresolved until post-contact evidence.
                        self.state = BallShotState.RIM_CONTACT
                        self.saw_rim_contact = True
                        self.evidence.append("Ball crossed rim contact zone")
                    else:
                        self.state = BallShotState.CROSSED_OUTSIDE
                        self.evidence.append("Ball crossed outside rim opening")

        below_threshold = max(
            rim_y + radius * self.below_margin_scale,
            rim_bottom + radius * 0.25,
        )
        below_rim = snapshot.y > below_threshold
        inside_net_channel = (
            horizontal_offset <= self.rim_inner_radius * self.net_channel_scale
        )
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
            if observed and below_rim and inside_net_channel:
                self.below_confirmation_count += 1
                self.miss_confirmation_count = 0
            elif observed:
                self.below_confirmation_count = 0

            clear_outside_below = observed and below_rim and not inside_net_channel
            if rebounding_up or outside_exit or clear_outside_below:
                self.miss_confirmation_count += 1

            if self.below_confirmation_count >= self.below_confirmation_frames:
                contact_make = self.state == BallShotState.RIM_CONTACT
                self._finish(
                    result="made",
                    confidence=0.90 if contact_make else 0.95,
                    evidence=(
                        "Ball observed below rim inside net channel after rim contact"
                        if contact_make
                        else "Ball observed below rim inside net channel"
                    ),
                    timestamp_ms=timestamp_ms,
                )
            elif self.miss_confirmation_count >= self.miss_confirmation_frames:
                self._finish(
                    result="missed",
                    confidence=0.85,
                    evidence="Ball clearly rebounded or exited after rim contact",
                    timestamp_ms=timestamp_ms,
                )

        elif self.state == BallShotState.CROSSED_OUTSIDE:
            if (
                observed
                and (
                    (below_rim and not inside_net_channel)
                    or rebounding_up
                    or outside_exit
                )
            ):
                self.miss_confirmation_count += 1
            if self.miss_confirmation_count >= self.miss_confirmation_frames:
                self._finish(
                    result="missed",
                    confidence=0.90,
                    evidence="Ball passed below rim outside opening",
                    timestamp_ms=timestamp_ms,
                )

        elif self.state == BallShotState.RIM_APPROACH:
            # Handles an airball when sparse frames do not capture the exact
            # plane crossing but the ball is visibly below and well outside.
            if (
                observed
                and below_rim
                and horizontal_offset
                > self.rim_inner_radius * self.rim_exit_channel_scale
            ):
                self.miss_confirmation_count += 1
            if self.miss_confirmation_count >= self.miss_confirmation_frames:
                self._finish(
                    result="missed",
                    confidence=0.80,
                    evidence="Ball continued below and outside the basket",
                    timestamp_ms=timestamp_ms,
                )

        return crossed

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
            self.evidence.append("Possible rim contact")
            self._recorded_rim_contact = True
            self.saw_rim_contact = True

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
            outcome_timestamp_ms=timestamp_ms,
            evidence=list(self.evidence),
            timeseries_summary={
                "ball_state": self.state.value,
                "saw_ball_above_rim": self.saw_ball_above_rim,
                "saw_rim_contact": self.saw_rim_contact,
                "crossing_xy": self.crossing_xy,
            },
        )
        return self.outcome

    def _result(
        self,
        tracking_status: BallTrackingStatus,
        released_this_frame: bool = False,
        crossed_rim_this_frame: bool = False,
    ) -> BallStateUpdate:
        return BallStateUpdate(
            state=self.state,
            tracking_status=tracking_status,
            outcome=self.outcome,
            released_this_frame=released_this_frame,
            crossed_rim_this_frame=crossed_rim_this_frame,
            rim_center_xy=self.rim_center,
            rim_inner_radius=self.rim_inner_radius,
            crossing_xy=self.crossing_xy,
        )

    def _remember_pose_phase(self, pose_phase: Optional[str]) -> None:
        if pose_phase is not None:
            self._previous_pose_phase = pose_phase
