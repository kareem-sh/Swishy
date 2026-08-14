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
        shot_type: Optional[str] = None,
    ) -> AnalysisResult:
        active: List[RuleResult] = []
        violations: List[RuleResult] = []
        passed = 0
        total = 0

        for rule_id, rule in self._rules.items():
            phases = rule.get("phases", [])
            if phase not in phases:
                continue
            if not self._applies_to(rule, shot_type):
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

    @staticmethod
    def _applies_to(rule: dict, shot_type: Optional[str]) -> bool:
        """Does this rule run for this kind of shot?

        A rule with no `shot_types` runs for every shot, which is what almost
        all of them want: a knee bend is a knee bend. A rule that names them
        runs only for those, so a mechanic that exists in one shot type can be
        assessed without inventing a verdict for a shot that never performs it.

        `shot_type is None` means the caller could not tell us -- the live path,
        where the classification is not settled until the attempt closes. A
        scoped rule is SKIPPED there rather than guessed at, because running a
        jump-shot rule on an unclassified attempt is exactly the kind of quiet
        assumption that turns "not observed" into a measurement.
        """
        allowed = rule.get("shot_types")
        if not allowed:
            return True
        return shot_type is not None and shot_type in allowed

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

        # Both release metrics are built from `wrist_y`, which is fabricated as
        # 0.0 when the visibility gate rejects the wrist -- and in hip-centred
        # world coordinates 0.0 reads as "the wrist is exactly at hip height",
        # a perfectly plausible number that nothing downstream can question.
        #
        # `release` is the worst phase for this: it is the fastest-moving and
        # most motion-blurred part of the shot, so it is where the gate is most
        # likely to reject. The failure would read as "released far too low"
        # and produce a whole coaching sentence out of a landmark nobody saw.
        #
        # `wrist_world_valid` exists precisely for this and is checked at every
        # other read site in the project. These two were the exceptions.
        if metric in ("release_height", "release_height_ratio"):
            if not features.wrist_world_valid:
                return None

        if metric == "release_height":
            # Body-relative: how far the hand clears the head. Both landmarks
            # come from the SAME frame, so the hip-centred origin cancels and
            # the result is a true body-relative height, scale-free and usable
            # with or without a recorded player height.
            if features.nose_y == 0:
                return None
            return features.wrist_y - features.nose_y

        if metric == "release_height_ratio":
            # Floor-relative release height as a fraction of standing height,
            # the representation the literature uses (SOURCES.md A2).
            #
            # THIS SUBTRACTION IS NOT SAME-FRAME, and that is why the ankle
            # reference has to come from image space.
            #
            # `wrist_y` is this frame's hip-relative wrist. `ankle_baseline_y`
            # is a hip-relative ankle height averaged while the player stood
            # still BEFORE the shot. Subtracting one from the other does not
            # cancel the origin, because the two are measured against the hip
            # at different moments -- and between those moments the hip itself
            # rose. The result is `arm_reach + leg_length`, with the jump term
            # missing entirely: structurally blind to the elevation, which is
            # the one thing a release height is supposed to capture.
            #
            # Measured consequence on salah_video: all five shots read 0.4-0.9
            # against a band starting at 1.02 and published values of
            # 1.12-1.17. The rule failed 100% of attempts and told every player
            # they release too low.
            #
            # `takeoff_ratio` is the same elevation the shot classifier uses,
            # measured in IMAGE space where whole-body translation is visible
            # at all, in body-height units. Adding it back restores the term
            # the world-space subtraction discards.
            # Added AFTER normalising, because `body_rise_ratio` is already a
            # fraction of the player's on-screen height -- the same unit
            # `normalized` produces. Converting it to metres first would need
            # `height_m`, which is None whenever no height was entered, and
            # would reintroduce a division this term does not require.
            ratio = self._player.normalized(
                features.wrist_y - features.ankle_baseline_y
            )
            if ratio is None:
                return None
            return ratio + max(0.0, features.body_rise_ratio or 0.0)

        if metric in ("takeoff_elevation", "landing_settle"):
            # How far the whole body left the floor, as a fraction of the
            # player's own on-screen height.
            #
            # IMAGE SPACE, deliberately. `ankle_rise` and
            # `vertical_displacement` below both ask this question of
            # hip-centred world landmarks, where the hip IS the origin and
            # whole-body translation is therefore invisible -- what they
            # actually return is the hip-to-ankle distance changing, i.e. the
            # legs folding. This is the same signal the shot classifier and the
            # phase refiner already trust to decide whether the feet left the
            # floor at all, so scoring against it keeps the three components
            # describing one event rather than three.
            #
            # Corroborated against A1 without any entered height. Measured
            # across 16 shots, peak elevation was 0.011-0.108 body heights for
            # set shots and 0.211 for the one labelled jump shot. Taking
            # nose-to-ankle as roughly 0.87 of stature, those land near 14 cm
            # and 33 cm for a ~1.8 m player -- against A1's published 15.3 cm
            # for free throws and 26.9-31.2 cm for jump shots. The agreement is
            # a check on the measurement, NOT a licence to report centimetres:
            # the conversion needs a stature we do not have, so the value stays
            # in body heights, which is what we actually observe.
            #
            # `landing_settle` reads the SAME signal at the other end of the
            # shot. Its zero is not a chosen threshold: `body_rise_ratio` is
            # measured against the player's own stance before this attempt, so
            # "back where you started" IS zero by construction. That is what
            # makes a band around it principled rather than fitted.
            return features.body_rise_ratio

        if metric == "wrist_height":
            return features.wrist_y

        if metric == "ankle_rise":
            return features.ankle_y_avg - features.ankle_baseline_y

        if metric == "vertical_displacement":
            # Reported in centimetres to match the literature. Used to tell a
            # set shot (~15 cm) from a jump shot (~27-31 cm).
            return (features.ankle_y_avg - features.ankle_baseline_y) * 100.0

        return None
