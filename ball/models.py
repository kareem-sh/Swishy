"""Phase 6 data models for ball tracking and shot outcome."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BallDetection:
    """Per-frame ball detection in image space."""

    center_xy: Tuple[float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    frame_index: int = 0
    timestamp_ms: int = 0


@dataclass
class BallSnapshot:
    """Ball state at one timestamp for time-series analysis."""

    timestamp_ms: int
    frame_index: int
    center_xy: Tuple[float, float]
    confidence: float
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    state: str = "unknown"  # in_hand | in_flight | at_rim | unknown


@dataclass
class BallTrajectory:
    """Fitted flight path for one shot attempt."""

    release_frame: Optional[int] = None
    apex_frame: Optional[int] = None
    entry_frame: Optional[int] = None
    entry_angle_deg: Optional[float] = None
    snapshots: List[BallSnapshot] = field(default_factory=list)


@dataclass
class ShotOutcome:
    """Make / miss / unknown result for one shot."""

    result: str = "unknown"  # made | missed | unknown
    confidence: float = 0.0
    release_frame: Optional[int] = None
    release_timestamp_ms: Optional[int] = None
    entry_frame: Optional[int] = None
    trajectory_apex_frame: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    timeseries_summary: Dict = field(default_factory=dict)
