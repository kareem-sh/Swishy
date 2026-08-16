"""Lightweight constant-velocity basketball tracker."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ball.models import BallDetection, BallSnapshot


class BallTracker:
    """Smooth detections and predict through short missed-detection gaps.

    The measurement receives most of the weight to avoid the visible lag caused
    by a moving average on a fast ball. Velocity is smoothed separately.
    """

    def __init__(
        self,
        max_gap_ms: int = 200,
        position_alpha: float = 0.80,
        velocity_alpha: float = 0.50,
    ):
        self.max_gap_ms = max(0, int(max_gap_ms))
        self.position_alpha = min(1.0, max(0.0, float(position_alpha)))
        self.velocity_alpha = min(1.0, max(0.0, float(velocity_alpha)))

        self.track_id = 0
        self.track_history: List[BallSnapshot] = []
        self.current_position: Optional[Tuple[float, float]] = None
        self.current_velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_detection_frame = -1
        self.last_detection_timestamp_ms: Optional[int] = None
        self.last_timestamp_ms: Optional[int] = None

    def update(
        self,
        detection: Optional[BallDetection],
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallSnapshot]:
        """Update state with a detection or a short constant-velocity prediction."""
        if detection is not None and detection.is_visual_observation:
            snapshot = self._update_from_detection(
                detection,
                frame_index,
                timestamp_ms,
            )
            self.track_history.append(snapshot)
            return snapshot

        snapshot = self._predict_missing(frame_index, timestamp_ms)
        if snapshot is not None:
            self.track_history.append(snapshot)
        return snapshot

    def _update_from_detection(
        self,
        detection: BallDetection,
        frame_index: int,
        timestamp_ms: int,
    ) -> BallSnapshot:
        measured_x, measured_y = detection.center_xy

        if self.current_position is None or self.last_timestamp_ms is None:
            smoothed = (measured_x, measured_y)
            velocity = (0.0, 0.0)
        else:
            dt = max((timestamp_ms - self.last_timestamp_ms) / 1000.0, 1e-3)
            predicted_x = self.current_position[0] + self.current_velocity[0] * dt
            predicted_y = self.current_position[1] + self.current_velocity[1] * dt
            alpha = self.position_alpha
            smoothed = (
                alpha * measured_x + (1.0 - alpha) * predicted_x,
                alpha * measured_y + (1.0 - alpha) * predicted_y,
            )

            measured_velocity = (
                (smoothed[0] - self.current_position[0]) / dt,
                (smoothed[1] - self.current_position[1]) / dt,
            )
            beta = self.velocity_alpha
            velocity = (
                beta * measured_velocity[0]
                + (1.0 - beta) * self.current_velocity[0],
                beta * measured_velocity[1]
                + (1.0 - beta) * self.current_velocity[1],
            )

        self.current_position = smoothed
        self.current_velocity = velocity
        self.last_detection_frame = frame_index
        self.last_detection_timestamp_ms = timestamp_ms
        self.last_timestamp_ms = timestamp_ms

        return BallSnapshot(
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            center_xy=smoothed,
            confidence=detection.confidence,
            velocity_xy=velocity,
            state="unknown",
            track_id=self.track_id,
            measurement_source=detection.measurement_source,
        )

    def _predict_missing(
        self,
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallSnapshot]:
        gap_ms = (
            timestamp_ms - self.last_detection_timestamp_ms
            if self.last_detection_timestamp_ms is not None
            else None
        )
        if (
            gap_ms is None
            or gap_ms <= 0
            or gap_ms > self.max_gap_ms
            or self.current_position is None
            or self.last_timestamp_ms is None
            or not self.track_history
        ):
            return None

        dt = max((timestamp_ms - self.last_timestamp_ms) / 1000.0, 0.0)
        predicted = (
            self.current_position[0] + self.current_velocity[0] * dt,
            self.current_position[1] + self.current_velocity[1] * dt,
        )
        self.current_position = predicted
        self.last_timestamp_ms = timestamp_ms

        return BallSnapshot(
            timestamp_ms=timestamp_ms,
            frame_index=frame_index,
            center_xy=predicted,
            confidence=self.track_history[-1].confidence * 0.65,
            velocity_xy=self.current_velocity,
            state="unknown",
            track_id=self.track_id,
            is_interpolated=True,
            measurement_source="predicted",
        )

    def reset(self) -> None:
        """Clear state for a new shot/session."""
        self.track_id += 1
        self.track_history = []
        self.current_position = None
        self.current_velocity = (0.0, 0.0)
        self.last_detection_frame = -1
        self.last_detection_timestamp_ms = None
        self.last_timestamp_ms = None

    def get_track(self) -> List[BallSnapshot]:
        return self.track_history.copy()

    def get_current_position(self) -> Optional[Tuple[float, float]]:
        return self.current_position
