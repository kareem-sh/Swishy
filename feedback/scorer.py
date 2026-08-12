"""Aggregate rule results across frames into a weighted shot score."""

from typing import Dict, List

from analysis.models import RuleResult
from feedback.models import ShotSummary
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameSnapshot

_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}


def _aggregate_rules(frames: List[FrameSnapshot]) -> Dict[str, RuleResult]:
    """One outcome per rule_id — failure on any frame counts as failed."""
    aggregated: Dict[str, RuleResult] = {}

    for snapshot in frames:
        if not snapshot.analysis:
            continue
        for rule in snapshot.analysis.active_rules:
            existing = aggregated.get(rule.rule_id)
            if existing is None:
                aggregated[rule.rule_id] = rule
                continue
            if not rule.passed and existing.passed:
                aggregated[rule.rule_id] = rule
            elif not rule.passed and not existing.passed:
                if _SEVERITY_RANK.get(rule.severity, 0) > _SEVERITY_RANK.get(existing.severity, 0):
                    aggregated[rule.rule_id] = rule

    return aggregated


def score_shot(
    frames: List[FrameSnapshot],
    shot_number: int,
    started_mid_phase: bool = False,
    ended_early: bool = False,
    entry_phase: str | None = None,
) -> ShotSummary:
    """Compute 0-100 score and collect pass/fail lists for one shot."""
    cfg = load_yaml("scoring.yaml")
    weights = cfg.get("severity_weights", {"error": 3, "warning": 2, "info": 1})
    min_frames = int(cfg.get("min_evaluated_rules", 1))

    rules = _aggregate_rules(frames)
    rule_list = list(rules.values())

    if not rule_list:
        return ShotSummary(
            shot_number=shot_number,
            score=0,
            passed_count=0,
            total_count=0,
            coaching_tips=["Not enough data to score this shot — stay in frame."],
            phases_seen=_phases_seen(frames),
            started_mid_phase=started_mid_phase,
            ended_early=ended_early,
            entry_phase=entry_phase,
        )

    total_weight = sum(float(weights.get(r.severity, 1)) for r in rule_list)
    earned_weight = sum(float(weights.get(r.severity, 1)) for r in rule_list if r.passed)

    score = round(100.0 * earned_weight / total_weight) if total_weight > 0 else 0
    passed = [r for r in rule_list if r.passed]
    violations = [r for r in rule_list if not r.passed]

    if len(rule_list) < min_frames:
        score = max(0, score - 10)

    return ShotSummary(
        shot_number=shot_number,
        score=int(score),
        passed_count=len(passed),
        total_count=len(rule_list),
        passed_rules=passed,
        violations=violations,
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
