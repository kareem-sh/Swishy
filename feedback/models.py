"""Shot scoring and coaching feedback data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from analysis.models import RuleResult

if TYPE_CHECKING:
    from ball.models import ShotOutcome


@dataclass
class ShotSummary:
    """Aggregated evaluation for one completed shot attempt."""

    shot_number: int
    score: int  # 0-100
    passed_count: int
    total_count: int
    passed_rules: List[RuleResult] = field(default_factory=list)
    violations: List[RuleResult] = field(default_factory=list)
    coaching_tips: List[str] = field(default_factory=list)
    phases_seen: List[str] = field(default_factory=list)
    started_mid_phase: bool = False
    ended_early: bool = False
    entry_phase: Optional[str] = None
    missing_phases: List[str] = field(default_factory=list)
    capture_note: str = ""
    next_rep_focus: List[str] = field(default_factory=list)
    practice_drills: List[str] = field(default_factory=list)
    performance_actions: List[str] = field(default_factory=list)
    outcome: Optional["ShotOutcome"] = None

    @property
    def grade(self) -> str:
        if self.score >= 90:
            return "Excellent"
        if self.score >= 75:
            return "Good"
        if self.score >= 60:
            return "Fair"
        return "Needs Work"
