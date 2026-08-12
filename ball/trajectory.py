"""Parabolic arc fitting and flight metrics (Phase 6d)."""

from typing import Optional, List
import numpy as np

# Try to import scipy, fall back to manual fitting if not available
try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False  # numpy.polyfit is enough for v1

from ball.models import BallSnapshot, BallTrajectory
from ball.timeseries import BallTimeSeriesBuffer


class TrajectoryAnalyzer:
    """Fit ball flight path and extract apex / entry angle proxy."""

    def __init__(self, config_name: str = "ball.yaml"):
        """Initialize trajectory analyzer with configuration."""
        from utils.config_loader import load_yaml

        self.config = load_yaml(config_name)
        self.fit_quality_threshold = self.config.get("fit_quality_threshold", 0.7)
        self.min_snapshots_for_fit = self.config.get("min_snapshots_for_fit", 5)
    def fit_trajectory(self, snapshots: List[BallSnapshot]) -> Optional[BallTrajectory]:
        """Fit parabolic arc to in-flight snapshots."""
        if len(snapshots) < self.min_snapshots_for_fit:
            return None

        # Extract y coordinates
        y_data = [s.y for s in snapshots]
        timestamps = [s.timestamp_ms for s in snapshots]

        # Normalize timestamps for numerical stability
        t0 = timestamps[0] / 1000.0
        t_norm = [(t / 1000.0 - t0) for t in timestamps]

        # Fit parabola: y = a*t^2 + b*t + c
        try:
            if SCIPY_AVAILABLE:
                # Use scipy for fitting
                p0 = [0.0, 0.0, y_data[0]]
                popt, pcov = curve_fit(self._parabola, t_norm, y_data, p0=p0)
                a, b, c = popt
                r_squared = self._calculate_r_squared(y_data, t_norm, popt)
            else:
                # Manual polynomial fit using numpy
                coeffs = np.polyfit(t_norm, y_data, 2)
                a, b, c = coeffs
                r_squared = self._calculate_r_squared_numpy(y_data, t_norm, coeffs)

            if r_squared < self.fit_quality_threshold:
                return None

            # Find apex (where derivative = 0)
            # y = a*t^2 + b*t + c, derivative: 2*a*t + b = 0
            if a != 0:
                t_apex_norm = -b / (2 * a)
                apex_time = t_apex_norm + t0
                apex_y = self._parabola(t_apex_norm, a, b, c)
                
                # Find apex x position (interpolate from data)
                apex_x = self._interpolate_x_at_time(t_apex_norm + t0, snapshots)
            else:
                apex_x, apex_y, apex_time = None, None, None

            trajectory = BallTrajectory(
                snapshots=snapshots,
                fit_params={'a': a, 'b': b, 'c': c},
                r_squared=r_squared,
                apex_x=apex_x,
                apex_y=apex_y,
                apex_time_ms=apex_time * 1000 if apex_time else None
            )
            
            return trajectory

        except Exception as e:
            print(f"Trajectory fit failed: {e}")
            return None

    def _parabola(self, t, a, b, c):
        """Parabola function: y = a*t^2 + b*t + c."""
        return a * np.array(t)**2 + b * np.array(t) + c

    def _calculate_r_squared(self, y_data, t_norm, popt):
        """Calculate R-squared using scipy fit results."""
        y_pred = self._parabola(t_norm, *popt)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _calculate_r_squared_numpy(self, y_data, t_norm, coeffs):
        """Calculate R-squared using numpy fit results."""
        y_pred = np.polyval(coeffs, t_norm)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _interpolate_x_at_time(self, time_sec: float, snapshots: List[BallSnapshot]) -> Optional[float]:
        """Interpolate x position at given time."""
        if len(snapshots) < 2:
            return None

        times = [s.timestamp_ms / 1000.0 for s in snapshots]
        
        for i in range(len(times) - 1):
            if times[i] <= time_sec <= times[i+1]:
                # Linear interpolation
                t0, t1 = times[i], times[i+1]
                x0, x1 = snapshots[i].x, snapshots[i+1].x
                if t1 == t0:
                    return x0
                return x0 + (x1 - x0) * (time_sec - t0) / (t1 - t0)
        
        return None

    def estimate_entry_angle(self, trajectory: BallTrajectory) -> Optional[float]:
        """Estimate entry angle at hoop plane (degrees)."""
        if trajectory is None or trajectory.fit_params is None:
            return None

        a, b, _ = trajectory.fit_params['a'], trajectory.fit_params['b'], trajectory.fit_params['c']
        
        # Derivative of parabola: dy/dt = 2*a*t + b
        # At rim crossing, we need the time when y reaches hoop level
        
        if not trajectory.snapshots:
            return None

        # Estimate rim level from last few snapshots (assuming they're near rim)
        last_y = np.mean([s.y for s in trajectory.snapshots[-3:]])
        
        # Find time when trajectory reaches this y level
        # y = a*t^2 + b*t + c = last_y
        # a*t^2 + b*t + (c - last_y) = 0
        c = trajectory.fit_params['c']
        
        discriminant = b**2 - 4*a*(c - last_y)
        if discriminant < 0:
            return None
        
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Choose the later time (ball going down)
        t_entry = max(t1, t2)
        
        # Derivative at entry
        dy_dt = 2*a*t_entry + b
        dx_dt = 1.0  # Approximate (normalized)
        
        angle_rad = np.arctan2(dy_dt, dx_dt)
        angle_deg = np.degrees(angle_rad)
        
        # Convert to entry angle from horizontal
        return 90 - abs(angle_deg)

    def analyze_shot_window(self, ball_buffer: BallTimeSeriesBuffer) -> Optional[BallTrajectory]:
        """Full trajectory analysis for one shot time window."""
        if len(ball_buffer) < self.min_snapshots_for_fit:
            return None

        # Get all snapshots from buffer
        snapshots = list(ball_buffer.buffer)
        
        # Filter to flight phase (after release, before rim)
        events = ball_buffer.detect_events()
        
        release_frame = events.get('release')
        rim_frame = events.get('rim_crossing')
        
        if release_frame is not None:
            # Filter snapshots after release
            release_snapshots = [s for s in snapshots if s.frame_index >= release_frame]
            
            if rim_frame is not None:
                # Filter until rim crossing
                flight_snapshots = [s for s in release_snapshots if s.frame_index <= rim_frame]
            else:
                flight_snapshots = release_snapshots
            
            return self.fit_trajectory(flight_snapshots)
        
        return self.fit_trajectory(snapshots)