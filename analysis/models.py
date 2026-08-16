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
        """Fraction of this rule's weight earned, tier only.

        The fallback for results that carry no numbers to grade against --
        `follow_through_hold` builds its result by hand, and older callers set
        an outcome without a band. `RuleResult.credit` refines NEEDS_WORK by
        HOW FAR outside the band the measurement fell; this is what it decays
        to when that is unknowable.
        """
        return {
            RuleOutcome.EXCELLENT: 1.0,
            RuleOutcome.GOOD: GOOD_CREDIT,
            RuleOutcome.NEEDS_WORK: 0.0,
        }[self]


# Credit for a measurement inside the acceptable band but outside the ideal.
# A shot with every rule rated GOOD therefore scores 72 rather than excellent.
GOOD_CREDIT = 0.72

# A missed band can retain limited credit so small errors decay smoothly rather
# than dropping the rule immediately to zero.
MISS_CREDIT_MAX = 0.45


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

    @property
    def credit(self) -> float:
        """Fraction of this rule's weight earned, graded by how far off it was.

        Three tiers, and inside the failing one a slope:

            in the ideal band          1.00
            in the acceptable band     GOOD_CREDIT
            outside it                 MISS_CREDIT_MAX, decaying to 0.00

        The decay is measured in UNITS OF THE RULE'S OWN BAND, not in degrees
        or metres. A rule is only as strict as its band is narrow, so a fixed
        penalty would punish a 5 deg overshoot on a 68 deg band exactly as hard
        as the same overshoot on a 20 deg one. Dividing by the band's own width
        makes "missed by a little" mean the same thing to every rule, whatever
        it measures and whatever unit it measures in.

        Miss by a full band-width or more and the credit is zero, which is
        where the old cliff still lives -- for gross deviation, where it
        belongs.
        """
        if self.outcome is not RuleOutcome.NEEDS_WORK:
            return self.outcome.credit
        excess = self._excess()
        span = self._span()
        if excess is None or span is None or span <= 0:
            # Nothing to grade against. Fall back to the tier, which for
            # NEEDS_WORK is 0.0 -- the strict reading, never a generous guess.
            return self.outcome.credit
        return MISS_CREDIT_MAX * max(0.0, 1.0 - excess / span)

    def _excess(self) -> Optional[float]:
        """How far outside the acceptable band the measurement fell."""
        if self.measured_value is None:
            return None
        if self.min_value is not None and self.measured_value < self.min_value:
            return self.min_value - self.measured_value
        if self.max_value is not None and self.measured_value > self.max_value:
            return self.measured_value - self.max_value
        return None

    def _span(self) -> Optional[float]:
        """The rule's own scale, for turning an excess into a fraction.

        Preference order matters. A two-sided band states its scale outright.
        A one-sided band (`follow_through_hold` has a floor and no ceiling)
        does not, so the distance from the bound to the ideal bound stands in
        for it -- that gap is the rule's own statement of what counts as a
        meaningful amount of this quantity.
        """
        if self.min_value is not None and self.max_value is not None:
            return self.max_value - self.min_value
        for bound, ideal in ((self.min_value, self.ideal_min),
                             (self.max_value, self.ideal_max)):
            if bound is not None and ideal is not None:
                gap = abs(ideal - bound)
                if gap > 0:
                    return gap
        return None


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
