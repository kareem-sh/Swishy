"""Projectile motion and make/miss probability (#11)."""

from typing import List, Optional

from ball.models import BallTrajectory
from physics.models import PhysicsTrajectory, ProjectileState, TrajectoryEstimate


class PhysicsTrajectoryEstimator:
    """Estimate ball trajectory using projectile physics equations."""

    def __init__(self):
        # TODO: load gravity, scale, and rim geometry from config/physics.yaml
        pass

    def fit_from_snapshots(self, snapshots: list) -> Optional[TrajectoryEstimate]:
        """Fit release velocity and arc from observed ball positions."""
        # TODO: parabolic least-squares fit in image or court plane
        # TODO: estimate initial velocity (vx0, vy0) at release
        pass

    def fit_from_trajectory(self, trajectory: BallTrajectory) -> Optional[TrajectoryEstimate]:
        """Convert Phase 6 BallTrajectory into physics estimate."""
        # TODO: reuse ball/trajectory.py output as input
        pass

    def simulate_flight(
        self,
        initial: ProjectileState,
        dt: float,
        steps: int,
    ) -> List[ProjectileState]:
        """Simulate projectile motion under constant gravity."""
        # TODO: Euler integration with g from config
        pass

    def estimate_make_probability(
        self,
        estimate: TrajectoryEstimate,
        hoop_center_xy: tuple,
        rim_radius_px: float,
    ) -> float:
        """Predict make probability from entry angle and rim offset."""
        # TODO: compare simulated rim crossing to hoop geometry
        pass

    def analyze(self, trajectory: BallTrajectory) -> Optional[PhysicsTrajectory]:
        """Full physics report: arc, entry angle, make probability."""
        # TODO: fit → simulate → score against hoop ROI
        # TODO: compare to optimal entry angle (e.g. 45–52°)
        pass

    def pixels_to_meters(self, distance_px: float, reference_scale: float) -> float:
        """Convert pixel distance to meters using court calibration."""
        # TODO: homography or known rim diameter calibration
        pass
