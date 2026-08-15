"""Detailed report data models for session and shot analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
from feedback.models import ShotSummary


@dataclass
class VisibilityGapNote:
    """Period when a landmark/angle could not be measured reliably."""

    body_part: str
    phase: str
    phase_label: str
    start_ms: int
    end_ms: int
    start_frame: int
    end_frame: int
    hold_frames_exceeded: int

    def summary(self) -> str:
        start = _gap_ts(self.start_ms)
        end = _gap_ts(self.end_ms)
        return (
            f"{start} -> {end} during {self.phase_label}: "
            f"could not reliably see the {self.body_part} "
            f"({self.hold_frames_exceeded} frames below confidence threshold)"
        )


def _gap_ts(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


@dataclass
class KeyFrame:
    """Annotated frame highlighting a phase or form issue."""

    frame_index: int
    timestamp_ms: int
    timestamp_label: str
    phase: str
    phase_label: str
    image_filename: str
    capture_reason: str
    issues: List[str] = field(default_factory=list)
    rule_ids: List[str] = field(default_factory=list)
    angles_summary: str = ""


@dataclass
class PhaseMoment:
    """When a phase occurred during the shot."""

    phase: str
    phase_label: str
    timestamp_ms: int
    timestamp_label: str
    frame_index: int


@dataclass
class RuleEvaluation:
    """Full rule result with coaching context for reports."""

    rule_id: str
    name: str
    passed: bool
    severity: str
    message: str
    rationale: str
    phase: str
    measured_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    key_frame_filename: Optional[str] = None


@dataclass
class DetailedShotReport:
    """Complete analysis for one shot attempt."""

    summary: ShotSummary
    phase_timeline: List[PhaseMoment] = field(default_factory=list)
    rule_evaluations: List[RuleEvaluation] = field(default_factory=list)
    key_frames: List[KeyFrame] = field(default_factory=list)
    visibility_gaps: List[VisibilityGapNote] = field(default_factory=list)
    frame_images: Dict[int, "np.ndarray"] = field(default_factory=dict)
    start_timestamp_ms: int = 0
    end_timestamp_ms: int = 0


@dataclass
class SessionReport:
    """Full form analysis for a video, live session, or image."""

    session_id: str
    source_type: str  # video | live | image
    source_name: str
    fps: float
    total_frames: int
    shots: List[DetailedShotReport] = field(default_factory=list)
    overall_score: Optional[int] = None
    overall_grade: str = "N/A"
    top_improvements: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    practice_plan: List[str] = field(default_factory=list)
    session_notes: List[str] = field(default_factory=list)
    output_path: Optional[str] = None

    @property
    def shot_count(self) -> int:
        return len(self.shots)
