"""Projectile motion and make/miss probability (#11)."""

from typing import List, Optional
import numpy as np

from ball.models import BallTrajectory
from physics.models import PhysicsTrajectory, ProjectileState, TrajectoryEstimate


class PhysicsTrajectoryEstimator:
    """Estimate ball trajectory using projectile physics equations."""

    def __init__(self, config_path: str = "config/physics.yaml"):
        """Initialize physics trajectory estimator."""
        self.config = self._load_config(config_path)
        self.gravity = self.config.get("gravity", 9.81)  # m/s^2
        self.scale = self.config.get("scale", 0.01)  # meters per pixel
        self.rim_radius_m = self.config.get("rim_radius_m", 0.23)  # meters
        self.optimal_entry_angle_min = self.config.get("optimal_entry_angle_min", 45)
        self.optimal_entry_angle_max = self.config.get("optimal_entry_angle_max", 52)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    def fit_from_snapshots(self, snapshots: list) -> Optional[TrajectoryEstimate]:
        """Fit release velocity and arc from observed ball positions."""
        if len(snapshots) < 5:
            return None

        # Convert pixels to meters
        x_m = [s.x * self.scale for s in snapshots]
        y_m = [s.y * self.scale for s in snapshots]
        t = [(s.timestamp_ms / 1000.0) for s in snapshots]
        t0 = t[0]

        # Normalize time
        t_norm = [ti - t0 for ti in t]

        # Fit x(t) = x0 + vx0 * t (constant velocity in x)
        # Fit y(t) = y0 + vy0 * t + 0.5 * g * t^2 (projectile motion)
        
        try:
            # Linear fit for x
            x_coeffs = np.polyfit(t_norm, x_m, 1)
            vx0 = x_coeffs[0]
            x0 = x_coeffs[1]

            # Quadratic fit for y
            y_coeffs = np.polyfit(t_norm, y_m, 2)
            a, b, c = y_coeffs
            
            # y = a*t^2 + b*t + c, where a = 0.5 * g (in image coordinates)
            # Note: gravity in image coordinates is positive (y increases downward)
            g_observed = 2 * a / self.scale  # Convert to m/s^2 if needed
            
            vy0 = b
            y0 = c

            # Calculate quality metrics
            y_pred = np.polyval(y_coeffs, t_norm)
            ss_res = np.sum((y_m - y_pred) ** 2)
            ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Find apex
            t_apex = -b / (2 * a) if a != 0 else 0
            apex_x = x0 + vx0 * t_apex
            apex_y = c - b**2 / (4 * a) if a != 0 else y0

            estimate = TrajectoryEstimate(
                vx0=vx0,
                vy0=vy0,
                x0=x0,
                y0=y0,
                t0=t0,
                gravity_estimate=g_observed,
                fit_quality=r_squared,
                apex_x=apex_x,
                apex_y=apex_y,
                apex_time=t0 + t_apex
            )
            
            return estimate

        except Exception as e:
            print(f"Physics fit failed: {e}")
            return None

    def fit_from_trajectory(self, trajectory: BallTrajectory) -> Optional[TrajectoryEstimate]:
        """Convert Phase 6 BallTrajectory into physics estimate."""
        if trajectory is None or not trajectory.snapshots:
            return None
        
        return self.fit_from_snapshots(trajectory.snapshots)

    def simulate_flight(
        self,
        initial: ProjectileState,
        dt: float,
        steps: int,
    ) -> List[ProjectileState]:
        """Simulate projectile motion under constant gravity."""
        states = []
        state = ProjectileState(
            x=initial.x,
            y=initial.y,
            vx=initial.vx,
            vy=initial.vy,
            t=initial.t
        )
        
        for _ in range(steps):
            # Update velocity (gravity in positive y direction)
            state.vy += self.gravity * self.scale * dt
            # Update position
            state.x += state.vx * dt
            state.y += state.vy * dt
            state.t += dt
            
            states.append(ProjectileState(
                x=state.x,
                y=state.y,
                vx=state.vx,
                vy=state.vy,
                t=state.t
            ))
        
        return states

    def estimate_make_probability(
        self,
        estimate: TrajectoryEstimate,
        hoop_center_xy: tuple,
        rim_radius_px: float,
    ) -> float:
        """Predict make probability from entry angle and rim offset."""
        if estimate is None:
            return 0.0

        # Convert hoop center to meters
        hoop_x = hoop_center_xy[0] * self.scale
        hoop_y = hoop_center_xy[1] * self.scale
        rim_radius_m = rim_radius_px * self.scale

        # Time to reach hoop x coordinate
        if abs(estimate.vx0) < 1e-6:
            return 0.0
        
        t_hoop = (hoop_x - estimate.x0) / estimate.vx0
        
        if t_hoop < 0:
            return 0.0

        # Position at hoop
        y_at_hoop = estimate.y0 + estimate.vy0 * t_hoop
        
        # Distance from hoop center
        y_error = abs(y_at_hoop - hoop_y)
        x_error = 0  # Assumes we're aligned with hoop x
        
        # Entry angle at hoop
        vy_at_hoop = estimate.vy0  # Constant acceleration not fully modeled here
        entry_angle_deg = np.degrees(np.arctan2(vy_at_hoop, estimate.vx0))
        entry_angle_deg = abs(entry_angle_deg)

        # Probability components
        # 1. Position accuracy
        max_error = rim_radius_m * 1.5
        if y_error > max_error:
            position_prob = 0.0
        else:
            position_prob = 1.0 - (y_error / max_error)
        
        # 2. Entry angle
        optimal_entry_angle = 48  # degrees from horizontal
        angle_error = abs(entry_angle_deg - optimal_entry_angle)
        if angle_error > 20:
            angle_prob = 0.0
        else:
            angle_prob = 1.0 - (angle_error / 20)
        
        # 3. Velocity (not too fast, not too slow)
        speed = np.sqrt(estimate.vx0**2 + estimate.vy0**2)
        optimal_speed = 6.0  # m/s (estimate)
        speed_error = abs(speed - optimal_speed)
        if speed_error > 3:
            speed_prob = 0.0
        else:
            speed_prob = 1.0 - (speed_error / 3)

        # Combined probability (weighted)
        probability = (
            0.5 * position_prob +
            0.3 * angle_prob +
            0.2 * speed_prob
        )
        
        return max(0.0, min(1.0, probability))

    def analyze(self, trajectory: BallTrajectory) -> Optional[PhysicsTrajectory]:
        """Full physics report: arc, entry angle, make probability."""
        if trajectory is None:
            return None

        estimate = self.fit_from_trajectory(trajectory)
        if estimate is None:
            return None

        # Get hoop position from trajectory (last snapshots)
        if not trajectory.snapshots:
            return None
        
        hoop_x_avg = np.mean([s.x for s in trajectory.snapshots[-3:]])
        hoop_y_avg = np.mean([s.y for s in trajectory.snapshots[-3:]])
        hoop_center = (hoop_x_avg, hoop_y_avg)

        # Estimate rim radius (from config or from trajectory)
        rim_radius_px = self.config.get("rim_radius_px", 30)

        # Calculate make probability
        make_prob = self.estimate_make_probability(estimate, hoop_center, rim_radius_px)

        # Simulate flight for visualization
        initial_state = ProjectileState(
            x=estimate.x0,
            y=estimate.y0,
            vx=estimate.vx0,
            vy=estimate.vy0,
            t=estimate.t0
        )
        
        # Simulate until ball reaches hoop level or 2 seconds
        dt = 0.01
        steps = int(2.0 / dt)
        simulated_states = self.simulate_flight(initial_state, dt, steps)

        # Calculate entry angle
        entry_angle = self._calculate_entry_angle(estimate, hoop_center)
        is_optimal = self.optimal_entry_angle_min <= entry_angle <= self.optimal_entry_angle_max

        physics_trajectory = PhysicsTrajectory(
            estimate=estimate,
            make_probability=make_prob,
            entry_angle=entry_angle,
            is_optimal_entry_angle=is_optimal,
            simulated_states=simulated_states,
            apex_x=estimate.apex_x,
            apex_y=estimate.apex_y,
            apex_time=estimate.apex_time
        )
        
        return physics_trajectory

    def _calculate_entry_angle(self, estimate: TrajectoryEstimate, hoop_center: tuple) -> float:
        """Calculate entry angle at hoop plane."""
        if estimate is None:
            return 0.0

        hoop_x_m = hoop_center[0] * self.scale
        t_hoop = (hoop_x_m - estimate.x0) / estimate.vx0 if estimate.vx0 != 0 else 0
        
        if t_hoop < 0:
            return 0.0

        vy_at_hoop = estimate.vy0  # Simple approximation
        vx_at_hoop = estimate.vx0
        
        entry_angle = np.degrees(np.arctan2(vy_at_hoop, vx_at_hoop))
        return abs(entry_angle)

    def pixels_to_meters(self, distance_px: float, reference_scale: float) -> float:
        """Convert pixel distance to meters using court calibration."""
        return distance_px * reference_scale