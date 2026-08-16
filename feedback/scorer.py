"""Aggregate rule results across frames into per-phase and overall shot scores."""

from typing import Dict, List

from analysis.engine import BiomechanicsEngine
from analysis.models import RuleOutcome, RuleResult
from feedback.models import PhaseScore, ShotSummary
from phase_detection.phases import ACTIVE_PHASES, PHASE_LABELS, PHASE_ORDER
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameSnapshot

_SEVERITY_RANK = {"error": 3, "warning": 2, "info": 1}
_OUTCOME_RANK = {
    RuleOutcome.NEEDS_WORK: 0,
    RuleOutcome.GOOD: 1,
    RuleOutcome.EXCELLENT: 2,
}


def _aggregate_policies() -> Dict[str, str]:
    """Which frame of a phase represents each rule. From biomechanics.yaml."""
    rules = (load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {}
    return {rid: str(r.get("aggregate", "worst")).lower()
            for rid, r in rules.items()}


def _rate_limits() -> Dict[str, float]:
    """The fastest each rule's metric can physically change, per SECOND.

    Per second, never per frame: this footage runs from 12 to 30 fps and
    includes slow motion, so a per-frame limit would be four different limits
    depending on the clip.

    A rule that declares no `max_rate` is not despiked at all, which is why
    adding this could not move a score that did not contain an impossible
    reading.
    """
    rules = (load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {}
    out: Dict[str, float] = {}
    for rid, r in rules.items():
        limit = r.get("max_rate")
        if limit is not None:
            out[rid] = float(limit)
    return out


def _despike(entries: List[tuple], max_rate: float) -> List[tuple]:
    """Drop lone frames that jump away from their neighbours and back.

    WHY THIS IS NOT AN OUTLIER FILTER
    ---------------------------------
    It removes nothing for being unusual. It removes frames that describe a
    movement no body can perform, and only when the frames on either side
    agree with each other about what was happening instead.

    Measured on video8_shot04: the shooting-side knee read 139.2, then 31.9,
    then 80.4 deg on three consecutive frames -- a swing of 107 deg and back in
    roughly 33 ms, about 3200 deg/s, with the hip failing on the same frame.
    That is a whole-frame pose failure, not a deep knee bend.

    It mattered because `min` and `max` seek the extreme value, so a single
    corrupt frame is exactly the frame they select. That phase scored 0 and the
    player was told to bend less deeply, from a frame in which the pose was
    wrong.

    A PLAIN RATE TEST IS NOT ENOUGH. That corrupt frame arrives after a 133 ms
    tracking gap -- the pose was lost for four frames and the first one back
    was wrong -- and across a gap that long a knee genuinely can travel 107
    degrees. "Degrees per second since the last frame" reads 806 deg/s and
    calls it plausible.

    What gives it away is the shape: the frames on either side agree with each
    other, only this one disagrees, and the next is back on the line. So the
    test is the distance from the straight line joining the neighbours against
    how far the joint could have strayed AND RETURNED, which is bounded by the
    SHORTER of the two gaps because the excursion had to reverse within it.

    A frame at the very start or end has only one neighbour and is kept: with
    nothing to corroborate it, dropping it would be a guess.

    `entries` is (timestamp_ms, RuleResult) in time order.
    """
    if len(entries) < 3:
        return entries

    values = [r.measured_value for _, r in entries]
    stamps = [ts for ts, _ in entries]
    kept: List[tuple] = [entries[0]]
    for i in range(1, len(entries) - 1):
        here, before, after = values[i], values[i - 1], values[i + 1]
        if here is not None and before is not None and after is not None:
            back_s = abs(stamps[i] - stamps[i - 1]) / 1000.0
            fwd_s = abs(stamps[i + 1] - stamps[i]) / 1000.0
            if back_s > 0 and fwd_s > 0:
                expected = before + (after - before) * (back_s / (back_s + fwd_s))
                if abs(here - expected) > max_rate * min(back_s, fwd_s):
                    continue
        kept.append(entries[i])
    kept.append(entries[-1])
    return kept


def _aggregate_rules(frames: List[FrameSnapshot]) -> Dict[str, RuleResult]:
    """One result per (phase, rule), chosen by that rule's `aggregate` policy.

    WHY THIS IS NOT ALWAYS "THE WORST FRAME"
    ----------------------------------------
    It used to be, for every rule, and for half of them that was the wrong
    question asked in a way that could not be noticed from the code.

    A rule describes one of two different things.

    CONTINUOUS rules ask "did this ever slip?" -- trunk lean, head stability.
    A posture that broke for a moment broke. The worst frame is the answer.

    EXTREMUM rules ask "how far did the movement GO?" -- how deep was the dip,
    how far did the arm extend. There is one frame that answers that, and it is
    not the worst one. Scoring the knee by its worst frame scores the moment
    the player had not yet bent, because "less bend" scores worse; the phase
    begins at the straightest knee by construction, so the worst frame is
    always the first one.

    Measured on salah_video before this was fixed: the loading phase contained
    knee angles from 91 deg (a good, deep load) up to 165 deg (standing), and
    the rule reported 165 and told the player to bend deeper -- on every shot,
    to a player whose actual load of 91-102 deg sits at or below the published
    preparatory norm of 107-116. The advice was not merely wrong, it was
    reversed.

    Policies, set per rule in biomechanics.yaml:
        min    the smallest measured value in the phase (deepest flexion)
        max    the largest (fullest extension, the held finish)
        worst  the lowest-scoring frame (continuous rules) -- the default,
               so a rule that does not declare a policy keeps the old
               behaviour rather than silently changing meaning.
    """
    policies = _aggregate_policies()
    limits = _rate_limits()

    # Gathered in time order first, because despiking needs each frame's
    # neighbours and the streaming comparison below cannot see them.
    #
    # Keyed by RULE, not by (phase, rule), and split into phases only after the
    # despike. A corrupt frame does not respect phase boundaries -- and worse,
    # it tends to CREATE one: `loading` ends at the smallest knee angle in the
    # shot, so a frame whose knee is wrong by 100 deg becomes the dip bottom and
    # therefore the last frame of its own phase. Despiking within the phase
    # could never see it, because the frame that would contradict it was pushed
    # into the next phase by the same corruption.
    series: Dict[str, List[tuple]] = {}
    for snapshot in frames:
        if not snapshot.analysis:
            continue
        for rule in snapshot.analysis.active_rules:
            series.setdefault(rule.rule_id, []).append(
                (snapshot.timestamp_ms, rule)
            )

    aggregated: Dict[str, RuleResult] = {}
    for rule_id, entries in series.items():
        limit = limits.get(rule_id)
        if limit is not None:
            entries = _despike(entries, limit)
        policy = policies.get(rule_id, "worst")
        for _, rule in entries:
            key = f"{rule.phase}:{rule_id}"
            chosen = aggregated.get(key)
            if chosen is None or _replaces(rule, chosen, policy):
                aggregated[key] = rule

    return aggregated


def _replaces(new: RuleResult, old: RuleResult, policy: str) -> bool:
    """Does `new` represent this phase better than `old`, under `policy`?"""
    if policy in ("min", "max"):
        # An extremum is only meaningful over frames that HAVE a value. A
        # frame the engine could not measure must never displace one it could,
        # or "not observed" quietly becomes the reported measurement.
        if new.measured_value is None:
            return False
        if old.measured_value is None:
            return True
        return (new.measured_value < old.measured_value if policy == "min"
                else new.measured_value > old.measured_value)

    new_rank = _OUTCOME_RANK.get(new.outcome, 0)
    old_rank = _OUTCOME_RANK.get(old.outcome, 0)
    if new_rank < old_rank:
        return True
    if new_rank == old_rank and not new.passed:
        return (_SEVERITY_RANK.get(new.severity, 0)
                > _SEVERITY_RANK.get(old.severity, 0))
    return False


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
        # None, not 0, when there is nothing here to score. See PhaseScore.
        score=_weighted_score(all_rules, weights) if scored else None,
        rules=scored,
        measured=[r for r in all_rules if not r.scored],
        strengths=_notes(spoken, RuleOutcome.EXCELLENT),
        refinements=_notes(spoken, RuleOutcome.GOOD),
        fixes=_notes(spoken, RuleOutcome.NEEDS_WORK),
    )


def _hold_rule(hold_s) -> RuleResult | None:
    """The follow-through hold, as a RuleResult, from a shot-level measurement.

    WHY THIS IS NOT AN ORDINARY RULE. `BiomechanicsEngine` evaluates one frame
    at a time and has no way to know when the release happened, so "seconds the
    arm stayed up after the ball left" cannot be a per-frame metric. It is
    measured once per shot by `phase_refiner.hold_duration_s` -- from the
    shooting event until the hand falls back below the shoulder, across every
    captured frame regardless of phase label -- and attached here.

    Everything else about it is an ordinary rule: bands, messages, severity and
    `scored` all come from biomechanics.yaml, so it is tuned in the same place
    as the rest and nothing about it is hardcoded.

    Returns None when the hand never came back down inside the clip. A
    recording that stopped early is not a perfect follow-through and not a
    dropped one -- it is not a measurement, and must not become a score.
    """
    if hold_s is None:
        return None
    cfg = ((load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {})
    rule = cfg.get("follow_through_hold")
    if not rule:
        return None

    min_v, max_v = rule.get("min"), rule.get("max")
    ideal_min, ideal_max = rule.get("ideal_min"), rule.get("ideal_max")
    outcome = BiomechanicsEngine._classify(hold_s, min_v, max_v, ideal_min, ideal_max)
    return RuleResult(
        rule_id="follow_through_hold",
        name=rule.get("name", "Holding the Finish"),
        passed=outcome is not RuleOutcome.NEEDS_WORK,
        severity=rule.get("severity", "warning"),
        message=BiomechanicsEngine._message(rule, outcome, hold_s, ideal_min, ideal_max),
        phase="follow_through",
        measured_value=hold_s,
        min_value=min_v,
        max_value=max_v,
        ideal_min=ideal_min,
        ideal_max=ideal_max,
        unit=rule.get("unit", "s"),
        scored=bool(rule.get("scored", True)),
        outcome=outcome,
        confidence=0.9,
    )


def _jump_release_timing_rule(
    offset_s: float | None,
    shot_type,
    confidence: float,
) -> RuleResult | None:
    """Build the jump-only release-vs-apex timing rule.

    This is shot-level for the same reason as the finish hold: one frame cannot
    know where the apex of the complete jump occurred. Positive values mean a
    late release after the apex plateau; negative values mean release while
    still rising.
    """
    type_name = getattr(shot_type, "value", shot_type)
    if offset_s is None or type_name != "jump_shot":
        return None

    cfg = ((load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {})
    rule = cfg.get("jump_release_timing")
    if not rule:
        return None

    min_v, max_v = rule.get("min"), rule.get("max")
    ideal_min, ideal_max = rule.get("ideal_min"), rule.get("ideal_max")
    outcome = BiomechanicsEngine._classify(
        offset_s, min_v, max_v, ideal_min, ideal_max
    )
    phases = rule.get("phases") or ["release"]
    return RuleResult(
        rule_id="jump_release_timing",
        name=rule.get("name", "Jump Release Timing"),
        passed=outcome is not RuleOutcome.NEEDS_WORK,
        severity=rule.get("severity", "warning"),
        message=BiomechanicsEngine._message(
            rule, outcome, offset_s, ideal_min, ideal_max
        ),
        phase=str(phases[0]),
        measured_value=offset_s,
        min_value=min_v,
        max_value=max_v,
        ideal_min=ideal_min,
        ideal_max=ideal_max,
        unit=rule.get("unit", "s"),
        scored=bool(rule.get("scored", True)),
        outcome=outcome,
        confidence=max(0.0, min(1.0, float(confidence))),
    )


def _expected_rules(phase: str, shot_type) -> List[str]:
    """Scored rules that SHOULD have produced a value for this phase.

    "Should" means the rule is scored and applies to this kind of shot. A rule
    scoped to jump shots is not expected in a set shot, so its absence there is
    the design working, not a measurement that failed.
    """
    cfg = (load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {}
    out = []
    for rid, rule in cfg.items():
        if phase not in (rule.get("phases") or []):
            continue
        if not rule.get("scored", True):
            continue
        allowed = rule.get("shot_types")
        if allowed and (shot_type is None or shot_type not in allowed):
            continue
        out.append(str(rule.get("name", rid)))
    return out


def _unmeasured_reason(phase: str, shot_type) -> str:
    """Why a phase the player performed carries no score.

    A phase that HAPPENED must never appear as a silent blank. Either it is
    scored, or the report says what could not be measured -- because at that
    point the limitation is in the footage, and the person filming is the only
    one who can fix it.
    """
    expected = _expected_rules(phase, shot_type)
    if not expected:
        # Nothing applies here by design. `jump` in a set shot is the case:
        # the player never left the floor, so there is no flight to assess and
        # no failure to report.
        return ""
    label = PHASE_LABELS.get(phase, phase.replace("_", " ").title())
    what = ", ".join(expected)
    return (
        f"{label} happened, but {what} could not be measured anywhere in it. "
        "That is a limit of this recording, not of the shot: film from the "
        "side with the whole body in frame and the shooting arm unobscured."
    )


def _build_phase_scores(
    rules: List[RuleResult],
    weights: dict,
    phases_present: List[str] | None = None,
    shot_type=None,
) -> List[PhaseScore]:
    """Group rules by phase and score each phase independently."""
    by_phase: Dict[str, List[RuleResult]] = {}
    for rule in rules:
        by_phase.setdefault(rule.phase, []).append(rule)

    # A phase the player performed but which produced NO rule at all never
    # reached `by_phase`, so it would simply vanish from the report. It is
    # added here with an explanation instead of being dropped.
    for phase in (phases_present or []):
        if phase in ACTIVE_PHASES:
            by_phase.setdefault(phase, [])

    ordered = [p for p in PHASE_ORDER if p in by_phase]
    ordered += [p for p in by_phase if p not in PHASE_ORDER]

    continuous = _whole_shot_rule_ids(rules)

    phase_scores = []
    for phase in ordered:
        score = _make_phase_score(
            phase,
            PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
            by_phase[phase],
            [r for r in by_phase[phase] if r.scored and r.rule_id not in continuous],
            weights,
        )
        if score.score is None:
            score.unmeasured_reason = _unmeasured_reason(phase, shot_type)
        phase_scores.append(score)

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
    shot_type=None,
    hold_s: float | None = None,
    jump_release_apex_offset_s: float | None = None,
    jump_release_timing_confidence: float = 0.75,
    shot_level_rules: List[RuleResult] | None = None,
) -> ShotSummary:
    """Compute per-phase scores and an overall 0-100 score for one shot."""
    cfg = load_yaml("scoring.yaml")
    weights = cfg.get("severity_weights", {"error": 3, "warning": 2, "info": 1})
    min_rules = int(cfg.get("min_evaluated_rules", 1))

    rule_list = list(_aggregate_rules(frames).values())

    # The follow-through hold is measured once per shot, not per frame, so it
    # joins the rule list here rather than coming out of the engine.
    hold_rule = _hold_rule(hold_s)
    if hold_rule is not None:
        rule_list.append(hold_rule)

    jump_timing_rule = _jump_release_timing_rule(
        jump_release_apex_offset_s,
        shot_type,
        jump_release_timing_confidence,
    )
    if jump_timing_rule is not None:
        rule_list.append(jump_timing_rule)

    # Complete-flight measurements (for example ball trajectory comparison)
    # are evaluated once per shot rather than once per pose frame. They join
    # the same RuleResult pipeline here so scoring, phase grouping, JSON, and
    # coaching messages do not need a parallel implementation.
    rule_list.extend(shot_level_rules or [])

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

    phase_scores = _build_phase_scores(
        rule_list, weights, _phases_seen(frames), shot_type
    )

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
