"""Parabolic arc fitting and flight metrics (Phase 6d)."""

from typing import Optional

from ball.models import BallSnapshot, BallTrajectory
from ball.timeseries import BallTimeSeriesBuffer


class TrajectoryAnalyzer:
    """Fit ball flight path and extract apex / entry angle proxy."""

    def __init__(self):
        # TODO: load fit parameters from config/ball.yaml
        pass

    def fit_trajectory(self, snapshots: list[BallSnapshot]) -> Optional[BallTrajectory]:
        """Fit parabolic arc to in-flight snapshots."""
        # TODO: least-squares parabola fit in image plane
        # TODO: identify apex frame (vy crosses zero)
        pass

    def estimate_entry_angle(self, trajectory: BallTrajectory) -> Optional[float]:
        """Estimate entry angle at hoop plane (degrees)."""
        # TODO: derivative of fitted arc near rim crossing
        pass

    def analyze_shot_window(self, ball_buffer: BallTimeSeriesBuffer) -> Optional[BallTrajectory]:
        """Full trajectory analysis for one shot time window."""
        # TODO: slice buffer from release to rim / end of post_shot window
        pass
