"""Ball position time-series buffer (Phase 6a)."""

from typing import List, Optional, Tuple

from ball.models import BallSnapshot


class BallTimeSeriesBuffer:
    """Ring buffer of ball snapshots for temporal event detection."""

    def __init__(self, max_frames: int = 300):
        self.max_frames = max_frames
        # TODO: initialize ring buffer storage

    def push(self, snapshot: BallSnapshot):
        """Append one ball snapshot."""
        # TODO: push into ring buffer, evict oldest when full
        pass

    def get_window(self, start_ms: int, end_ms: int) -> List[BallSnapshot]:
        """Return snapshots between start_ms and end_ms (inclusive)."""
        # TODO: filter buffer by timestamp
        pass

    def compute_velocity(self, snapshot: BallSnapshot) -> Tuple[float, float]:
        """Estimate smoothed vx, vy in px/s from recent history."""
        # TODO: finite difference or Savitzky–Golay smoothing
        pass

    def detect_events(self) -> dict:
        """Detect release, apex, and rim_crossing events from the buffer."""
        # TODO: return dict with event names -> frame_index / timestamp_ms
        pass

    def clear(self):
        """Empty the buffer."""
        # TODO: reset storage
        pass
