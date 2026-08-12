"""Ball position time-series buffer (Phase 6a)."""

from typing import List, Optional, Tuple, Dict
from collections import deque
import numpy as np

from ball.models import BallSnapshot


class BallTimeSeriesBuffer:
    """Ring buffer of ball snapshots for temporal event detection."""

    def __init__(self, max_frames: int = 300):
        self.max_frames = max_frames
        self.buffer = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.frame_indices = deque(maxlen=max_frames)

    def push(self, snapshot: BallSnapshot):
        """Append one ball snapshot."""
        if snapshot is None:
            return
        
        self.buffer.append(snapshot)
        self.timestamps.append(snapshot.timestamp_ms)
        self.frame_indices.append(snapshot.frame_index)

    def get_window(self, start_ms: int, end_ms: int) -> List[BallSnapshot]:
        """Return snapshots between start_ms and end_ms (inclusive)."""
        if not self.buffer:
            return []

        result = []
        for snapshot in self.buffer:
            if start_ms <= snapshot.timestamp_ms <= end_ms:
                result.append(snapshot)
        return result

    def compute_velocity(self, snapshot: BallSnapshot) -> Tuple[float, float]:
        """Estimate smoothed vx, vy in px/s from recent history."""
        if len(self.buffer) < 3:
            return (0.0, 0.0)

        # Get recent snapshots before the current one
        recent = [s for s in self.buffer if s.timestamp_ms < snapshot.timestamp_ms]
        if len(recent) < 2:
            return (0.0, 0.0)

        # Use last 5 snapshots for smoothing
        recent = recent[-5:]
        
        if len(recent) < 2:
            return (0.0, 0.0)

        # Calculate velocities between consecutive snapshots
        velocities_x = []
        velocities_y = []
        
        for i in range(len(recent) - 1):
            dt = (recent[i+1].timestamp_ms - recent[i].timestamp_ms) / 1000.0  # seconds
            if dt == 0:
                continue
            vx = (recent[i+1].x - recent[i].x) / dt
            vy = (recent[i+1].y - recent[i].y) / dt
            velocities_x.append(vx)
            velocities_y.append(vy)

        if not velocities_x:
            return (0.0, 0.0)

        # Return average velocity
        return (np.mean(velocities_x), np.mean(velocities_y))

    def detect_events(self) -> Dict[str, int]:
        """Detect release, apex, and rim_crossing events from the buffer."""
        if len(self.buffer) < 5:
            return {}

        events = {}
        
        # Detect release: sudden increase in ball-wrist distance or velocity spike
        release_idx = self._detect_release()
        if release_idx is not None:
            events['release'] = release_idx

        # Detect apex: y velocity crosses from positive to negative
        apex_idx = self._detect_apex()
        if apex_idx is not None:
            events['apex'] = apex_idx

        # Detect rim crossing: ball enters hoop region
        rim_idx = self._detect_rim_crossing()
        if rim_idx is not None:
            events['rim_crossing'] = rim_idx

        return events

    def _detect_release(self) -> Optional[int]:
        """Detect release event from velocity pattern."""
        if len(self.buffer) < 5:
            return None

        velocities = []
        for i in range(1, len(self.buffer)):
            dt = (self.buffer[i].timestamp_ms - self.buffer[i-1].timestamp_ms) / 1000.0
            if dt == 0:
                continue
            vx = (self.buffer[i].x - self.buffer[i-1].x) / dt
            vy = (self.buffer[i].y - self.buffer[i-1].y) / dt
            velocities.append((vx, vy, i))

        if len(velocities) < 3:
            return None

        # Look for velocity spike (release)
        for i in range(1, len(velocities) - 1):
            _, _, idx = velocities[i]
            prev_vx, prev_vy, _ = velocities[i-1]
            vx, vy, _ = velocities[i]
            
            speed = np.sqrt(vx**2 + vy**2)
            prev_speed = np.sqrt(prev_vx**2 + prev_vy**2)
            
            if speed > prev_speed * 3 and speed > 100:  # Threshold for release
                return self.buffer[idx].frame_index

        return None

    def _detect_apex(self) -> Optional[int]:
        """Detect apex where vertical velocity crosses zero."""
        if len(self.buffer) < 5:
            return None

        # Compute velocities
        velocities = []
        for i in range(1, len(self.buffer)):
            dt = (self.buffer[i].timestamp_ms - self.buffer[i-1].timestamp_ms) / 1000.0
            if dt == 0:
                continue
            vy = (self.buffer[i].y - self.buffer[i-1].y) / dt
            velocities.append((vy, i))

        if len(velocities) < 3:
            return None

        # Find where vy changes from positive to negative (upward to downward)
        for i in range(1, len(velocities)):
            vy_prev, _ = velocities[i-1]
            vy_curr, idx = velocities[i]
            
            if vy_prev > 0 and vy_curr < 0:
                return self.buffer[idx].frame_index

        return None

    def _detect_rim_crossing(self) -> Optional[int]:
        """Detect rim crossing event (simplified)."""
        # This would require hoop ROI configuration
        # Simplified: look for sudden drop in y position
        if len(self.buffer) < 5:
            return None

        for i in range(2, len(self.buffer)):
            if (self.buffer[i].y - self.buffer[i-2].y) > 50:  # Sudden drop threshold
                return self.buffer[i].frame_index

        return None

    def clear(self):
        """Empty the buffer."""
        self.buffer.clear()
        self.timestamps.clear()
        self.frame_indices.clear()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]