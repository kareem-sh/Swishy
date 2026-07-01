"""Align ball release with body pose release phase (Phase 6b)."""

from typing import Optional

from ball.models import BallSnapshot
from ball.timeseries import BallTimeSeriesBuffer


class ReleaseSync:
    """Fuse ball time-series with wrist landmarks and body phase."""

    def __init__(self):
        # TODO: load sync thresholds from config/ball.yaml
        pass

    def find_release_frame(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int],
        wrist_positions: list,
    ) -> Optional[int]:
        """Return best-estimate release frame index."""
        # TODO: match ball leaving hand (ball–wrist distance spike)
        # TODO: align with body FSM release phase within ±N frames
        pass

    def ball_wrist_distance(
        self,
        ball: BallSnapshot,
        wrist_xy: tuple,
    ) -> float:
        """Pixel distance between ball center and shooting wrist."""
        # TODO: euclidean distance in image space
        pass
