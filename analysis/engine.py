"""
Configurable biomechanical rule engine.

Rules are defined in config/biomechanics.yaml and evaluated only
during their relevant shot phases.
"""

from typing import List, Optional

from analysis.models import AnalysisResult, RuleResult
from phase_detection.features import KinematicFeatures
from utils.config_loader import load_yaml


class BiomechanicsEngine:
    """Evaluate shooting form rules for the current phase."""

    def __init__(self):
        cfg = load_yaml("biomechanics.yaml")
        self._rules = cfg.get("rules", {})

    def evaluate(
        self,
        phase: str,
        angles: dict,
        features: KinematicFeatures,
        shooting_side: str,
    ) -> AnalysisResult:
        active: List[RuleResult] = []
        violations: List[RuleResult] = []
        passed = 0
        total = 0

        for rule_id, rule in self._rules.items():
            phases = rule.get("phases", [])
            if phase not in phases:
                continue

            result = self._evaluate_rule(rule_id, rule, angles, features, shooting_side, phase)
            if result is None:
                continue

            total += 1
            active.append(result)
            if result.passed:
                passed += 1
            else:
                violations.append(result)

        return AnalysisResult(
            phase=phase,
            active_rules=active,
            violations=violations,
            passed_count=passed,
            total_count=total,
        )

    def _evaluate_rule(
        self,
        rule_id: str,
        rule: dict,
        angles: dict,
        features: KinematicFeatures,
        shooting_side: str,
        phase: str,
    ) -> Optional[RuleResult]:
        metric = rule.get("metric")
        min_val = rule.get("min")
        max_val = rule.get("max")
        severity = rule.get("severity", "warning")
        name = rule.get("name", rule_id)
        message_pass = rule.get("message_pass", f"Good {name}")
        message_fail = rule.get("message_fail", f"Adjust {name}")

        measured = self._measure(metric, angles, features, shooting_side)
        if measured is None:
            return None

        passed = True
        if min_val is not None and measured < min_val:
            passed = False
        if max_val is not None and measured > max_val:
            passed = False

        return RuleResult(
            rule_id=rule_id,
            name=name,
            passed=passed,
            severity=severity,
            message=message_pass if passed else message_fail,
            phase=phase,
            measured_value=measured,
            min_value=min_val,
            max_value=max_val,
            confidence=0.9,
        )

    def _measure(
        self,
        metric: str,
        angles: dict,
        features: KinematicFeatures,
        shooting_side: str,
    ) -> Optional[float]:
        side = shooting_side

        angle_metrics = {
            "knee": f"{side}_knee",
            "hip": f"{side}_hip",
            "elbow": f"{side}_elbow",
            "shoulder": f"{side}_shoulder",
            "index_align": f"{side}_index_align",
            "trunk": "trunk",
        }

        if metric in angle_metrics:
            key = angle_metrics[metric]
            if key not in angles or not angles[key].is_valid:
                return None
            return angles[key].degrees

        if metric == "head_velocity":
            return abs(features.nose_velocity_y)

        if metric == "release_height":
            if features.nose_y == 0:
                return None
            return features.wrist_y - features.nose_y

        if metric == "wrist_height":
            return features.wrist_y

        if metric == "ankle_rise":
            return features.ankle_y_avg - features.ankle_baseline_y

        return None
