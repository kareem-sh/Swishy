"""Data models for biomechanical rule evaluation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class RuleOutcome(str, Enum):
    """Three-tier result for one rule on one frame.

    Binary pass/fail cannot say "this is already good, here is how to make it
    excellent" -- which is most of what coaching actually is. Each rule has an
    acceptable band (min..max) and, optionally, an ideal band inside it.

        outside min..max      -> NEEDS_WORK  (say what to change)
        inside min..max       -> GOOD        (affirm, then refine)
        inside ideal band     -> EXCELLENT   (affirm, change nothing)
    """

    NEEDS_WORK = "needs_work"
    GOOD = "good"
    EXCELLENT = "excellent"

    @property
    def credit(self) -> float:
        """Fraction of this rule's weight earned toward the score."""
        return {
            RuleOutcome.EXCELLENT: 1.0,
            RuleOutcome.GOOD: 0.75,
            RuleOutcome.NEEDS_WORK: 0.0,
        }[self]


@dataclass
class RuleResult:
    rule_id: str
    name: str
    passed: bool
    severity: str  # info | warning | error
    message: str
    phase: str
    measured_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    confidence: float = 1.0
    outcome: RuleOutcome = RuleOutcome.NEEDS_WORK
    ideal_min: Optional[float] = None
    ideal_max: Optional[float] = None
    unit: str = "°"
    # False for metrics we measure and display but deliberately do not score,
    # because the evidence or the measurement is not strong enough. See
    # resources.md Part E.
    scored: bool = True

    def __post_init__(self):
        # `passed` and `outcome` must never disagree. Callers that predate the
        # three-tier model set only `passed`; without this, such a result would
        # carry passed=True yet score zero credit, silently.
        if not self.passed:
            self.outcome = RuleOutcome.NEEDS_WORK
        elif self.outcome is RuleOutcome.NEEDS_WORK:
            self.outcome = RuleOutcome.EXCELLENT

    @property
    def is_excellent(self) -> bool:
        return self.outcome is RuleOutcome.EXCELLENT


@dataclass
class AnalysisResult:
    phase: str
    active_rules: List[RuleResult] = field(default_factory=list)
    violations: List[RuleResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0

    @property
    def all_passed(self) -> bool:
        return self.total_count > 0 and self.passed_count == self.total_count
