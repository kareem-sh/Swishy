"""Align ball release with body pose release phase (Phase 6b)."""

from typing import Optional, List
import numpy as np

from ball.models import BallSnapshot
from ball.timeseries import BallTimeSeriesBuffer


class ReleaseSync:
    """Fuse ball time-series with wrist landmarks and body phase."""

    def __init__(self, config_path: str = "config/ball.yaml"):
        """Initialize release sync with configuration."""
        self.config = self._load_config(config_path)
        self.max_frame_offset = self.config.get("release_max_offset", 5)
        self.distance_threshold = self.config.get("wrist_distance_threshold", 50)
        self.velocity_threshold = self.config.get("release_velocity_threshold", 100)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

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

        # Align with body release if available
        if body_release_frame is not None and ball_release_idx is not None:
            offset = abs(ball_release_idx - body_release_frame)
            if offset <= self.max_frame_offset:
                return ball_release_idx
            else:
                # Use body release frame if within buffer range
                return body_release_frame

        return ball_release_idx

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