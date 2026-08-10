"""
Configurable biomechanical rule engine.

Rules are defined in config/biomechanics.yaml and evaluated only
during their relevant shot phases.
"""

from typing import List, Optional

from analysis.models import AnalysisResult, RuleOutcome, RuleResult
from phase_detection.features import KinematicFeatures
from player.profile import PlayerProfile
from utils.config_loader import load_yaml


class BiomechanicsEngine:
    """Evaluate shooting form rules for the current phase."""

    def __init__(self, player: Optional[PlayerProfile] = None):
        cfg = load_yaml("biomechanics.yaml")
        self._rules = cfg.get("rules", {})
        # Height is optional. Without it, height-dependent metrics return None
        # and their rules are skipped rather than scored against a guess.
        self._player = player or PlayerProfile()

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
        ideal_min = rule.get("ideal_min")
        ideal_max = rule.get("ideal_max")
        severity = rule.get("severity", "warning")
        name = rule.get("name", rule_id)

        measured = self._measure(metric, angles, features, shooting_side)
        if measured is None:
            return None

        outcome = self._classify(measured, min_val, max_val, ideal_min, ideal_max)
        message = self._message(rule, outcome, measured, ideal_min, ideal_max)

        return RuleResult(
            rule_id=rule_id,
            name=name,
            passed=outcome is not RuleOutcome.NEEDS_WORK,
            severity=severity,
            message=message,
            phase=phase,
            measured_value=measured,
            min_value=min_val,
            max_value=max_val,
            ideal_min=ideal_min,
            ideal_max=ideal_max,
            unit=rule.get("unit", "°"),
            scored=bool(rule.get("scored", True)),
            outcome=outcome,
            confidence=0.9,
        )

    @staticmethod
    def _classify(
        measured: float,
        min_val,
        max_val,
        ideal_min,
        ideal_max,
    ) -> RuleOutcome:
        """Place a measurement into the needs-work / good / excellent bands."""
        if min_val is not None and measured < min_val:
            return RuleOutcome.NEEDS_WORK
        if max_val is not None and measured > max_val:
            return RuleOutcome.NEEDS_WORK

        # Inside the acceptable band. Without an ideal band declared, being
        # acceptable IS the target -- do not invent a stricter one.
        if ideal_min is None and ideal_max is None:
            return RuleOutcome.EXCELLENT

        if ideal_min is not None and measured < ideal_min:
            return RuleOutcome.GOOD
        if ideal_max is not None and measured > ideal_max:
            return RuleOutcome.GOOD
        return RuleOutcome.EXCELLENT

    @staticmethod
    def _message(
        rule: dict,
        outcome: RuleOutcome,
        measured: float,
        ideal_min,
        ideal_max,
    ) -> str:
        """Pick the message for this outcome.

        The GOOD tier is directional: a value below the ideal band needs the
        opposite correction from one above it, so telling the player merely
        "refine this" would be useless or actively wrong.
        """
        name = rule.get("name", "this")

        if outcome is RuleOutcome.EXCELLENT:
            return rule.get("message_excellent", f"{name} is on target.")

        if outcome is RuleOutcome.NEEDS_WORK:
            if ideal_min is not None and measured < (rule.get("min") or ideal_min):
                return rule.get(
                    "message_low", rule.get("message_fail", f"Adjust {name}.")
                )
            return rule.get(
                "message_high", rule.get("message_fail", f"Adjust {name}.")
            )

        # GOOD — affirm first, then give one directional refinement.
        if ideal_min is not None and measured < ideal_min:
            return rule.get("refine_low", rule.get("message_excellent", f"{name} is good."))
        return rule.get("refine_high", rule.get("message_excellent", f"{name} is good."))

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
            # Body-relative: how far the hand clears the head. This is already
            # scale-free, so it works for every player with or without a
            # recorded height.
            if features.nose_y == 0:
                return None
            return features.wrist_y - features.nose_y

        if metric == "release_height_ratio":
            # Floor-relative release height as a fraction of standing height —
            # the representation used in the literature (Cabarkapa et al. 2023,
            # resources.md A2), where it was the only variable that separated
            # proficient from non-proficient free-throw shooters (p=0.010).
            #
            # Requires a user-provided height. Returns None otherwise, which
            # makes the engine skip the rule instead of inventing a value.
            return self._player.normalized(
                features.wrist_y - features.ankle_baseline_y
            )

        if metric == "wrist_height":
            return features.wrist_y

        if metric == "ankle_rise":
            return features.ankle_y_avg - features.ankle_baseline_y

        if metric == "vertical_displacement":
            # Reported in centimetres to match the literature. Used to tell a
            # set shot (~15 cm) from a jump shot (~27-31 cm).
            return (features.ankle_y_avg - features.ankle_baseline_y) * 100.0

        return None
