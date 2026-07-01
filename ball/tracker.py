"""Multi-frame ball tracking (Phase 6a)."""

from typing import List, Optional

import numpy as np

from ball.models import BallDetection, BallSnapshot


class BallTracker:
    """Maintain ball identity and smooth position across frames."""

    def __init__(self):
        # TODO: optional Kalman filter or ByteTrack-style association
        pass

    def update(
        self,
        detection: Optional[BallDetection],
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallSnapshot]:
        """Update tracker state; return smoothed snapshot or None."""
        # TODO: associate detection with track ID
        # TODO: interpolate when detection is missing for 1–2 frames
        pass

    def reset(self):
        """Clear track state (new shot or new session)."""
        # TODO: reset internal buffers and track ID
        pass

    def get_track(self) -> List[BallSnapshot]:
        """Return full track history for current shot."""
        # TODO: expose buffered snapshots
        pass
