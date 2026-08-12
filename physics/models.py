"""Data models for physics-based trajectory estimation (#11)."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TrajectoryEstimate:
    """Projectile parameters estimated from ball flight."""

    release_position_xy: Optional[Tuple[float, float]] = None
    release_velocity_xy: Optional[Tuple[float, float]] = None
    apex_xy: Optional[Tuple[float, float]] = None
    entry_angle_deg: Optional[float] = None
    flight_time_s: Optional[float] = None
    snapshots_used: int = 0


@dataclass
class PhysicsTrajectory:
    """Full physics analysis for one shot."""

    estimate: TrajectoryEstimate
    make_probability: Optional[float] = None
    optimal_entry_angle_deg: Optional[float] = None
    arc_height_px: Optional[float] = None
    evidence: List[str] = field(default_factory=list)


@dataclass
class ProjectileState:
    """State vector for simulation (x, y, vx, vy)."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    t: float = 0.0
