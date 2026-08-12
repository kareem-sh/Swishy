"""Data models for biomechanical rule evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional


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
