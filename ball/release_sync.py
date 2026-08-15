"""Align ball release with body pose release phase (Phase 6b)."""

from typing import Optional, List
import numpy as np

from ball.models import BallSnapshot
from ball.timeseries import BallTimeSeriesBuffer


class ReleaseSync:
    """Fuse ball time-series with wrist landmarks and body phase."""

    def __init__(self, config_name: str = "ball.yaml"):
        """Initialize release sync with configuration."""
        from utils.config_loader import load_yaml

        self.config = load_yaml(config_name)
        sync = self.config.get("release_sync", {})
        self.max_time_offset_s = max(
            0.0,
            float(
                sync.get(
                    "max_body_ball_time_delta_s",
                    self.config.get("release_max_offset_s", 0.10),
                )
            )
        )
        self.distance_threshold = float(
            sync.get(
                "ball_wrist_release_distance_px",
                self.config.get("wrist_distance_threshold", 50),
            )
        )
        self.velocity_threshold = float(
            self.config.get("release_velocity_threshold", 100)
        )

    def find_release_frame(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int],
        wrist_positions: List[tuple],
    ) -> Optional[int]:
        """Return best-estimate release frame index."""
        if len(ball_buffer) < 3 or len(wrist_positions) < 3:
            return None

        # Method 1: Ball-wrist distance spike
        ball_release_idx = self._detect_by_distance(ball_buffer, wrist_positions)

        if ball_release_idx is None:
            # Method 2: Velocity-based detection
            ball_release_idx = self._detect_by_velocity(ball_buffer)

        # Align by video time, never by frame count. Frame indices are only
        # identifiers used to locate the corresponding timestamps.
        if body_release_frame is not None and ball_release_idx is not None:
            ball_timestamp_ms = self._timestamp_for_frame(
                ball_buffer, ball_release_idx
            )
            body_timestamp_ms = self._timestamp_for_frame(
                ball_buffer, body_release_frame
            )
            if ball_timestamp_ms is None or body_timestamp_ms is None:
                return body_release_frame
            offset_s = abs(ball_timestamp_ms - body_timestamp_ms) / 1000.0
            if offset_s <= self.max_time_offset_s:
                return ball_release_idx
            return body_release_frame

        return ball_release_idx

    @staticmethod
    def _timestamp_for_frame(
        ball_buffer: BallTimeSeriesBuffer, frame_index: int
    ) -> Optional[int]:
        for snapshot in ball_buffer.buffer:
            if snapshot.frame_index == frame_index:
                return snapshot.timestamp_ms
        return None

    def _detect_by_distance(self, buffer: BallTimeSeriesBuffer, wrist_positions: List[tuple]) -> Optional[int]:
        """Detect release by ball-wrist distance spike."""
        if len(buffer) < 3 or len(wrist_positions) < 3:
            return None

        # Need to align timestamps or frame indices
        # Assume wrist_positions correspond to buffer frames
        distances = []
        
        for i, snapshot in enumerate(buffer.buffer):
            if i < len(wrist_positions):
                wrist_xy = wrist_positions[i]
                if wrist_xy is not None:
                    dist = self.ball_wrist_distance(snapshot, wrist_xy)
                    distances.append((dist, snapshot.frame_index))

        if len(distances) < 3:
            return None

        # Find spike in distance
        for i in range(1, len(distances) - 1):
            prev_dist, _ = distances[i-1]
            curr_dist, frame_idx = distances[i]
            next_dist, _ = distances[i+1]
            
            if curr_dist > self.distance_threshold:
                if curr_dist > prev_dist * 1.5 and curr_dist > next_dist * 1.5:
                    return frame_idx

        return None

    def _detect_by_velocity(self, buffer: BallTimeSeriesBuffer) -> Optional[int]:
        """Detect release by velocity spike."""
        if len(buffer) < 5:
            return None

        velocities = []
        for i in range(1, len(buffer)):
            dt = (buffer.buffer[i].timestamp_ms - buffer.buffer[i-1].timestamp_ms) / 1000.0
            if dt == 0:
                continue
            vx = (buffer.buffer[i].x - buffer.buffer[i-1].x) / dt
            vy = (buffer.buffer[i].y - buffer.buffer[i-1].y) / dt
            speed = np.sqrt(vx**2 + vy**2)
            velocities.append((speed, buffer.buffer[i].frame_index))

        if len(velocities) < 3:
            return None

        # Find velocity spike
        for i in range(1, len(velocities) - 1):
            prev_speed, _ = velocities[i-1]
            curr_speed, frame_idx = velocities[i]
            next_speed, _ = velocities[i+1]
            
            if curr_speed > self.velocity_threshold:
                if curr_speed > prev_speed * 2 and curr_speed > next_speed * 2:
                    return frame_idx

        return None

    def ball_wrist_distance(
        self,
        ball: BallSnapshot,
        wrist_xy: tuple,
    ) -> float:
        """Pixel distance between ball center and shooting wrist."""
        if ball is None or wrist_xy is None:
            return float('inf')
        
        dx = ball.x - wrist_xy[0]
        dy = ball.y - wrist_xy[1]
        return np.sqrt(dx**2 + dy**2)
