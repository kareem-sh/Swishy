"""Aggregate rule results across frames into per-phase and overall shot scores."""

from typing import Dict, List

from analysis.models import RuleOutcome, RuleResult
from feedback.models import PhaseScore, ShotSummary
from phase_detection.phases import PHASE_LABELS, PHASE_ORDER
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameSnapshot

_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}
_OUTCOME_RANK = {
    RuleOutcome.NEEDS_WORK: 0,
    RuleOutcome.GOOD: 1,
    RuleOutcome.EXCELLENT: 2,
}


def _aggregate_rules(frames: List[FrameSnapshot]) -> Dict[str, RuleResult]:
    """One outcome per (phase, rule) — the worst outcome seen wins.

    Keyed by phase as well as rule so a rule evaluated in several phases
    reports separately in each, rather than one phase silently overwriting
    another's result.
    """
    aggregated: Dict[str, RuleResult] = {}

    for snapshot in frames:
        if not snapshot.analysis:
            continue
        for rule in snapshot.analysis.active_rules:
            key = f"{rule.phase}:{rule.rule_id}"
            existing = aggregated.get(key)
            if existing is None:
                aggregated[key] = rule
                continue

            new_rank = _OUTCOME_RANK.get(rule.outcome, 0)
            old_rank = _OUTCOME_RANK.get(existing.outcome, 0)
            if new_rank < old_rank:
                aggregated[key] = rule
            elif new_rank == old_rank and not rule.passed:
                if _SEVERITY_RANK.get(rule.severity, 0) > _SEVERITY_RANK.get(
                    existing.severity, 0
                ):
                    aggregated[key] = rule

    return aggregated


def _weighted_score(rules: List[RuleResult], weights: dict) -> int:
    """0-100 from partial credit: excellent 1.0, good 0.75, needs work 0.0."""
    scored = [r for r in rules if r.scored]
    total = sum(float(weights.get(r.severity, 1)) for r in scored)
    if total <= 0:
        return 0
    earned = sum(
        float(weights.get(r.severity, 1)) * r.outcome.credit for r in scored
    )
    return int(round(100.0 * earned / total))


WHOLE_SHOT_PHASE = "whole_shot"
WHOLE_SHOT_LABEL = "Through the Whole Shot"


def _whole_shot_rule_ids(rules: List[RuleResult]) -> set:
    """Rule ids that were evaluated in more than one phase.

    Posture and head stability are checked continuously, so they produce an
    identical sentence in every phase they run in. Printed per phase that
    reads as a stuck record -- one real clip repeated the same line four
    times. Which rules behave this way is discovered from the data, not from
    a hardcoded list, so adding a continuous rule needs no change here.
    """
    phases_by_rule: Dict[str, set] = {}
    for rule in rules:
        phases_by_rule.setdefault(rule.rule_id, set()).add(rule.phase)
    return {rid for rid, phases in phases_by_rule.items() if len(phases) > 1}


def _notes(rules: List[RuleResult], outcome: RuleOutcome) -> List[str]:
    """Messages for one outcome tier, in order, without repeats."""
    seen, out = set(), []
    for rule in rules:
        if rule.outcome is outcome and rule.message not in seen:
            seen.add(rule.message)
            out.append(rule.message)
    return out


def _make_phase_score(phase: str, label: str, all_rules: List[RuleResult],
                      spoken: List[RuleResult], weights: dict) -> PhaseScore:
    """Score from every rule; speak only the ones handed in as `spoken`.

    The split matters. A continuously-checked rule genuinely applies to this
    phase, so it belongs in the phase's score. Its wording belongs somewhere
    else, said once.
    """
    scored = [r for r in all_rules if r.scored]
    return PhaseScore(
        phase=phase,
        label=label,
        score=_weighted_score(all_rules, weights),
        rules=scored,
        measured=[r for r in all_rules if not r.scored],
        strengths=_notes(spoken, RuleOutcome.EXCELLENT),
        refinements=_notes(spoken, RuleOutcome.GOOD),
        fixes=_notes(spoken, RuleOutcome.NEEDS_WORK),
    )


def _build_phase_scores(
    rules: List[RuleResult],
    weights: dict,
) -> List[PhaseScore]:
    """Group rules by phase and score each phase independently."""
    by_phase: Dict[str, List[RuleResult]] = {}
    for rule in rules:
        by_phase.setdefault(rule.phase, []).append(rule)

    ordered = [p for p in PHASE_ORDER if p in by_phase]
    ordered += [p for p in by_phase if p not in PHASE_ORDER]

    continuous = _whole_shot_rule_ids(rules)

    phase_scores = [
        _make_phase_score(
            phase,
            PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
            by_phase[phase],
            [r for r in by_phase[phase] if r.scored and r.rule_id not in continuous],
            weights,
        )
        for phase in ordered
    ]

    # The continuous rules, gathered once. Worst outcome across the phases
    # wins, so a posture that slipped anywhere is reported as slipping.
    if continuous:
        worst: Dict[str, RuleResult] = {}
        for rule in rules:
            if rule.rule_id not in continuous or not rule.scored:
                continue
            held = worst.get(rule.rule_id)
            if held is None or _OUTCOME_RANK.get(rule.outcome, 0) < _OUTCOME_RANK.get(
                held.outcome, 0
            ):
                worst[rule.rule_id] = rule
        gathered = list(worst.values())
        if gathered:
            phase_scores.append(
                _make_phase_score(
                    WHOLE_SHOT_PHASE, WHOLE_SHOT_LABEL, gathered, gathered, weights
                )
            )

    return phase_scores


def score_shot(
    frames: List[FrameSnapshot],
    shot_number: int,
    started_mid_phase: bool = False,
    ended_early: bool = False,
    entry_phase: str | None = None,
) -> ShotSummary:
    """Compute per-phase scores and an overall 0-100 score for one shot."""
    cfg = load_yaml("scoring.yaml")
    weights = cfg.get("severity_weights", {"error": 3, "warning": 2, "info": 1})
    min_rules = int(cfg.get("min_evaluated_rules", 1))

    rule_list = list(_aggregate_rules(frames).values())

    if not rule_list:
        return ShotSummary(
            shot_number=shot_number,
            score=0,
            passed_count=0,
            total_count=0,
            coaching_tips=["Not enough of the shot was visible to score it. "
                           "Keep your whole body in the frame."],
            phases_seen=_phases_seen(frames),
            started_mid_phase=started_mid_phase,
            ended_early=ended_early,
            entry_phase=entry_phase,
        )

    phase_scores = _build_phase_scores(rule_list, weights)

    # Overall score is computed across every scored rule, not as a mean of
    # phase scores: averaging phase means would let a phase carrying one rule
    # outweigh a phase carrying five.
    score = _weighted_score(rule_list, weights)

    scored_rules = [r for r in rule_list if r.scored]
    passed = [r for r in scored_rules if r.passed]
    violations = [r for r in scored_rules if not r.passed]

    if len(scored_rules) < min_rules:
        score = max(0, score - 10)

    return ShotSummary(
        shot_number=shot_number,
        score=int(score),
        passed_count=len(passed),
        total_count=len(scored_rules),
        passed_rules=passed,
        violations=violations,
        phase_scores=phase_scores,
        phases_seen=_phases_seen(frames),
        started_mid_phase=started_mid_phase,
        ended_early=ended_early,
        entry_phase=entry_phase,
    )


def _phases_seen(frames: List[FrameSnapshot]) -> List[str]:
    seen = []
    for f in frames:
        if f.phase not in seen:
            seen.append(f.phase)
    return seen
