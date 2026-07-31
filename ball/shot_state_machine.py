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

    def update(
        self,
        ball_detection: Optional[BallDetection],
        ball_snapshot: Optional[BallSnapshot],
        rim_detection: Optional[RimDetection],
        wrist_xy: Optional[Tuple[float, float]],
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

        if not self._rim_locked:
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

        if self.state == BallShotState.WAITING and wrist_distance is not None:
            in_hand_limit = self.release_distance_px
            if pose_phase in {"loading", "knee_flexion", "ball_lift"}:
                in_hand_limit *= 1.5
            if wrist_distance <= in_hand_limit:
                self.state = BallShotState.IN_HAND

        if self.state in {BallShotState.WAITING, BallShotState.IN_HAND}:
            if self._release_confirmed(
                pose_phase=pose_phase,
                wrist_distance=wrist_distance,
                speed=speed,
            ):
                self.state = BallShotState.RELEASED
                self.release_timestamp_ms = timestamp_ms
                self.release_frame = ball_snapshot.frame_index
                self.evidence.append("Ball release confirmed")
                released_this_frame = True

        if self.state in _ACTIVE_OUTCOME_STATES:
            if self.state == BallShotState.RELEASED:
                if vy < -self.ascending_velocity_px_s:
                    self.state = BallShotState.ASCENDING
                elif vy > self.descending_velocity_px_s:
                    self.state = BallShotState.DESCENDING
                    self._rim_locked = self.rim_center is not None
            elif self.state == BallShotState.ASCENDING:
                previous_vy = (
                    self.previous_snapshot.velocity_xy[1]
                    if self.previous_snapshot is not None
                    else vy
                )
                if previous_vy < 0 and vy >= self.descending_velocity_px_s:
                    self.state = BallShotState.DESCENDING
                    self._rim_locked = self.rim_center is not None
                    self.evidence.append("Ball passed trajectory apex")

            crossed_this_frame = self._process_rim_geometry(
                snapshot=ball_snapshot,
                observed=observed,
                vertical_velocity=vy,
                timestamp_ms=timestamp_ms,
            )

        if self.outcome is None:
            self._check_timeouts(timestamp_ms, tracking_status)

        self.previous_snapshot = ball_snapshot
        if observed:
            self.previous_observed_snapshot = ball_snapshot
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
        self._rim_samples.append((float(rim.x), float(rim.y), inner_radius))
        samples = np.asarray(self._rim_samples, dtype=np.float64)
        self.rim_center = (
            float(np.median(samples[:, 0])),
            float(np.median(samples[:, 1])),
        )
        self.rim_inner_radius = float(np.median(samples[:, 2]))

    def _release_confirmed(
        self,
        pose_phase: Optional[str],
        wrist_distance: Optional[float],
        speed: float,
    ) -> bool:
        pose_release = pose_phase in _RELEASE_PHASES
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
            pose_release and (separating or ball_fast or far_from_wrist)
        ) or (separating and ball_fast)

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
        distance = math.hypot(snapshot.x - rim_x, snapshot.y - rim_y)
        near_rim = distance <= self.rim_inner_radius * self.approach_scale

        if near_rim and self.state in _FLIGHT_STATES:
            self.state = BallShotState.RIM_APPROACH
            self._rim_locked = True

        above_margin = radius * self.above_margin_scale
        if observed and near_rim and snapshot.y < rim_y - above_margin:
            self.saw_ball_above_rim = True

        self._record_possible_rim_contact(snapshot, observed, radius)

        crossed = False
        previous = self.previous_snapshot
        if previous is not None and self.state == BallShotState.RIM_APPROACH:
            downward_crossing = previous.y < rim_y <= snapshot.y
            crossing_supported = observed or not previous.is_interpolated
            if (
                downward_crossing
                and crossing_supported
                and self.saw_ball_above_rim
                and vertical_velocity >= 0
            ):
                crossing_x = self._interpolate_crossing_x(previous, snapshot, rim_y)
                if crossing_x is not None:
                    crossed = True
                    self.crossing_xy = (crossing_x, rim_y)
                    self.entry_frame = snapshot.frame_index
                    clearance = max(
                        self.rim_inner_radius - radius,
                        self.rim_inner_radius * 0.20,
                    )
                    if abs(crossing_x - rim_x) <= clearance:
                        self.state = BallShotState.CROSSED_INSIDE
                        self.saw_inside_crossing = True
                        self.evidence.append("Ball crossed inside rim opening")
                    else:
                        self.state = BallShotState.CROSSED_OUTSIDE
                        self.evidence.append("Ball crossed outside rim opening")

        below_rim = snapshot.y > rim_y + radius * self.below_margin_scale
        horizontal_offset = abs(snapshot.x - rim_x)
        inside_net_channel = (
            horizontal_offset <= self.rim_inner_radius * self.net_channel_scale
        )

        if self.state == BallShotState.CROSSED_INSIDE:
            if observed and below_rim and inside_net_channel:
                self.below_confirmation_count += 1
            elif observed and not below_rim:
                self.below_confirmation_count = 0

            bounced_away = observed and (
                snapshot.y < rim_y - above_margin
                or horizontal_offset > self.rim_inner_radius * 1.5
            )
            if bounced_away:
                self.miss_confirmation_count += 1

            if self.below_confirmation_count >= self.below_confirmation_frames:
                self._finish(
                    result="made",
                    confidence=0.95,
                    evidence="Ball observed below rim inside net channel",
                    timestamp_ms=timestamp_ms,
                )
            elif self.miss_confirmation_count >= self.miss_confirmation_frames:
                self._finish(
                    result="missed",
                    confidence=0.85,
                    evidence="Ball bounced away after entering rim area",
                    timestamp_ms=timestamp_ms,
                )

        elif self.state == BallShotState.CROSSED_OUTSIDE:
            if observed and below_rim and not inside_net_channel:
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
                and horizontal_offset > self.rim_inner_radius * 1.5
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

    @staticmethod
    def _interpolate_crossing_x(
        previous: BallSnapshot,
        current: BallSnapshot,
        rim_y: float,
    ) -> Optional[float]:
        dy = current.y - previous.y
        if abs(dy) < 1e-6:
            return None
        ratio = (rim_y - previous.y) / dy
        if not 0.0 <= ratio <= 1.0:
            return None
        return previous.x + ratio * (current.x - previous.x)

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

        if tracking_status == BallTrackingStatus.OBSERVED:
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
