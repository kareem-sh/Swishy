"""Shot scoring and coaching feedback data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

from analysis.models import RuleResult
from shots.types import RejectionReason, ShotClassification, ShotType

if TYPE_CHECKING:
    from ball.models import ShotOutcome


@dataclass
class PhaseScore:
    """Score and notes for one phase of a single shot.

    Notes are split three ways so the player always hears what is already
    working, not only what is wrong:

        strengths   - already on target, change nothing
        refinements - good, with one specific way to make it better
        fixes       - outside the acceptable range, needs a change
    """

    phase: str
    label: str
    # 0-100, or None when the phase carries nothing scoreable.
    #
    # None is NOT zero. A phase whose rules are all unscored (or which has no
    # rules at all) has not been failed -- it has not been assessed. Emitting 0
    # made `jump` read as a total failure on every shot the moment its last
    # scored rule was withdrawn, which is the same class of error as letting an
    # unobserved landmark report 0.0.
    score: Optional[int]
    rules: List[RuleResult] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    refinements: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)
    measured: List[RuleResult] = field(default_factory=list)  # displayed, not scored

    @property
    def evaluated(self) -> bool:
        return bool(self.rules)

    @property
    def grade(self) -> str:
        if self.score is None:
            return "not scored"
        if self.score >= 90:
            return "Excellent"
        if self.score >= 75:
            return "Good"
        if self.score >= 60:
            return "Fair"
        return "Needs Work"


@dataclass
class ShotSummary:
    """Aggregated evaluation for one completed shot attempt."""

    shot_number: int
    # None means REJECTED. A rejection is not a score of zero: it carries no
    # phase scores and no coaching feedback, because no analyser ran.
    score: Optional[int]
    passed_count: int
    total_count: int
    shot_type: Optional[ShotType] = None
    classification: Optional[ShotClassification] = None
    rejection: Optional[RejectionReason] = None
    passed_rules: List[RuleResult] = field(default_factory=list)
    violations: List[RuleResult] = field(default_factory=list)
    phase_scores: List["PhaseScore"] = field(default_factory=list)
    coaching_tips: List[str] = field(default_factory=list)
    phases_seen: List[str] = field(default_factory=list)
    started_mid_phase: bool = False
    ended_early: bool = False
    entry_phase: Optional[str] = None
    missing_phases: List[str] = field(default_factory=list)
    capture_note: str = ""
    # What to work on next, and the drills for it. `performance_actions` used
    # to sit here too, holding `next_rep_focus` with a "Fix X:" prefix on each
    # line -- it was written on every shot and read by nothing.
    next_rep_focus: List[str] = field(default_factory=list)
    practice_drills: List[str] = field(default_factory=list)
    outcome: Optional["ShotOutcome"] = None
    # Where this attempt sits in the video, from real frame timestamps.
    # None means the caller did not record it -- never a placeholder.
    start_timestamp_ms: Optional[int] = None
    end_timestamp_ms: Optional[int] = None

    @property
    def is_rejected(self) -> bool:
        return self.rejection is not None or self.score is None

    @property
    def grade(self) -> str:
        if self.score is None:
            return "Not Scored"
        if self.score >= 90:
            return "Excellent"
        if self.score >= 75:
            return "Good"
        if self.score >= 60:
            return "Fair"
        return "Needs Work"
