"""Every rule, on one shot, phase by phase -- including the ones that did not run.

    venv/Scripts/python.exe scripts/inspect_shot.py assets/videos/video8.mov
    venv/Scripts/python.exe scripts/inspect_shot.py video.mp4 --shot 3
    venv/Scripts/python.exe scripts/inspect_shot.py video.mp4 --all

The coaching report answers "what should this player change". This answers a
different question -- "what did the system actually measure, and how did that
become a number" -- and it is the view you need when a score looks wrong.

WHY THE RULES THAT DID NOT RUN ARE PRINTED TOO
----------------------------------------------
A missing rule is the most common reason a score surprises someone, and it is
invisible in the report by construction: a rule that could not be measured
contributes nothing and says nothing. But "contributed nothing" has three very
different causes, and only one of them is a problem:

    not applicable   the rule is scoped to another shot type -- a jump rule in
                     a set shot is the design working, not a failure
    phase absent     the phase itself never happened in this shot
    NOT MEASURED     the phase happened and the rule could not read it. THIS
                     is a fact about the footage, and it is the one worth
                     acting on

Collapsing those three into a blank is how a limitation gets read as a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from analysis.models import RuleOutcome, RuleResult  # noqa: E402
from feedback.models import PhaseScore, ShotSummary  # noqa: E402
from feedback.scorer import WHOLE_SHOT_PHASE  # noqa: E402
from phase_detection.phases import PHASE_ORDER  # noqa: E402
from scripts.coach_report import analyze_video  # noqa: E402
from utils.config_loader import load_yaml  # noqa: E402

WEIGHTS = {"error": 3, "warning": 2, "info": 1}

MARK = {
    RuleOutcome.EXCELLENT: "[ON TARGET]",
    RuleOutcome.GOOD: "[REFINE]   ",
    RuleOutcome.NEEDS_WORK: "[CHANGE]   ",
}


def _band(r: RuleResult) -> str:
    lo = "-inf" if r.min_value is None else f"{r.min_value:g}"
    hi = "+inf" if r.max_value is None else f"{r.max_value:g}"
    out = f"{lo}..{hi}"
    if r.ideal_min is not None or r.ideal_max is not None:
        ilo = "-inf" if r.ideal_min is None else f"{r.ideal_min:g}"
        ihi = "+inf" if r.ideal_max is None else f"{r.ideal_max:g}"
        out += f"  ideal {ilo}..{ihi}"
    return out


def _value(r: RuleResult) -> str:
    if r.measured_value is None:
        return "not measured"
    return f"{r.measured_value:.3f}{r.unit}"


def _rule_line(r: RuleResult, cfg: dict) -> List[str]:
    spec = cfg.get(r.rule_id, {})
    w = WEIGHTS.get(r.severity, 1)
    lines = [
        f"    {MARK[r.outcome]} {r.name}  ({r.rule_id})",
        f"        measured   {_value(r)}"
        f"        aggregate: {spec.get('aggregate', 'worst')}",
        f"        band       {_band(r)}",
    ]
    if r.scored:
        lines.append(
            f"        scoring    severity {r.severity} = weight {w}"
            f"   x credit {r.credit:.2f}"
            f"   = {w * r.credit:.2f} of {w}"
        )
    else:
        lines.append("        scoring    NOT SCORED -- measured and displayed only")
    lines.append(f"        says       {r.message}")
    return lines


def _phase_block(ps: PhaseScore, cfg: dict) -> List[str]:
    out: List[str] = []
    head = f"  {ps.label}"
    if ps.score is None:
        head += "   [no score]"
    else:
        head += f"   {ps.score}/100  ({ps.grade})"
    out.append(head)
    out.append("  " + "-" * 74)

    if ps.unmeasured_reason:
        out.append(f"    ! {ps.unmeasured_reason}")

    for r in ps.rules:
        out += _rule_line(r, cfg)
    for r in ps.measured:
        out += _rule_line(r, cfg)

    if ps.rules:
        tw = sum(WEIGHTS.get(r.severity, 1) for r in ps.rules)
        te = sum(WEIGHTS.get(r.severity, 1) * r.credit for r in ps.rules)
        out.append(f"    => phase score  {te:.2f} / {tw} = "
                   f"{100.0 * te / tw:.0f}" if tw else "")
    out.append("")
    return out


# Metrics that cannot be computed without a player height. Naming them matters:
# "could not be read on any frame" is true of these and blames the footage,
# when the actual cause is an empty field the user can fill in.
HEIGHT_DEPENDENT = {"release_height_ratio"}


def _absent(summary: ShotSummary, cfg: dict, has_height: bool) -> List[str]:
    """Rules that never produced a result, and which of the reasons why."""
    ran = {r.rule_id for ps in summary.phase_scores for r in (ps.rules + ps.measured)}
    phases_scored = {ps.phase for ps in summary.phase_scores}
    shot_type = summary.shot_type.value if summary.shot_type else None

    out: List[str] = ["RULES THAT PRODUCED NOTHING", "-" * 76]
    any_row = False
    for rid, spec in cfg.items():
        if rid in ran:
            continue
        any_row = True
        phases = spec.get("phases") or []
        allowed = spec.get("shot_types")
        name = spec.get("name", rid)
        if allowed and (shot_type is None or shot_type not in allowed):
            why = f"not applicable: scoped to {allowed}, this shot is {shot_type}"
        elif spec.get("metric") in HEIGHT_DEPENDENT and not has_height:
            why = ("no player height given, so this metric is skipped rather "
                   "than estimated -- pass --height-cm")
        elif not any(p in phases_scored for p in phases):
            why = f"phase absent: {'/'.join(phases)} not present in this shot"
        else:
            why = ("NOT MEASURED: the phase happened and the metric "
                   f"'{spec.get('metric')}' could not be read on any frame in it")
        out.append(f"  {name:<34s} {why}")
    if not any_row:
        out.append("  (none -- every rule in the config produced a result)")
    out.append("")
    return out


def report(summary: ShotSummary, has_height: bool = False) -> str:
    cfg = (load_yaml("biomechanics.yaml") or {}).get("rules", {}) or {}
    out: List[str] = []

    out.append("=" * 76)
    kind = summary.shot_type.value if summary.shot_type else "unclassified"
    out.append(f"SHOT {summary.shot_number}   {kind}"
               + (f"   score {summary.score}/100" if summary.score is not None
                  else "   REJECTED"))
    if summary.classification is not None:
        out.append(f"  confidence {summary.classification.confidence:.2f}")
        for e in summary.classification.evidence:
            out.append(f"    - {e}")
    if summary.hold_duration_s is not None:
        out.append(f"  follow-through hold {summary.hold_duration_s:.2f}s")
    out.append(f"  phases seen: {', '.join(summary.phases_seen)}")
    out.append("=" * 76)
    out.append("")

    order = {p: i for i, p in enumerate(PHASE_ORDER)}
    ordered = sorted(
        summary.phase_scores,
        key=lambda ps: order.get(ps.phase, 99 if ps.phase == WHOLE_SHOT_PHASE else 98),
    )
    for ps in ordered:
        out += _phase_block(ps, cfg)

    out += _absent(summary, cfg, has_height)

    scored = [r for ps in summary.phase_scores for r in ps.rules]
    tw = sum(WEIGHTS.get(r.severity, 1) for r in scored)
    te = sum(WEIGHTS.get(r.severity, 1) * r.credit for r in scored)
    out.append("HOW THE OVERALL SCORE WAS BUILT")
    out.append("-" * 76)
    out.append(f"  {len(scored)} scored rule(s) across all phases")
    out.append(f"  total weight   {tw}")
    out.append(f"  earned         {te:.2f}")
    out.append(f"  score          100 x {te:.2f} / {tw} = "
               f"{round(100.0 * te / tw) if tw else 0}")
    out.append("  (the overall score is computed across every scored rule "
               "directly, NOT")
    out.append("   as a mean of the phase scores -- averaging phase means "
               "would let a")
    out.append("   phase carrying one rule outweigh a phase carrying five)")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--shot", type=int, default=1)
    ap.add_argument("--all", action="store_true", help="every shot in the clip")
    ap.add_argument("--height-cm", type=float, default=None)
    args = ap.parse_args()

    run = analyze_video(args.video, height_cm=args.height_cm, enable_ball=False)
    if run.is_rejected:
        print(f"rejected: {run.rejection.value}\n{run.rejection_detail}")
        return 1
    if not run.shots:
        print("no shot found")
        return 1

    wanted = run.shots if args.all else [
        s for s in run.shots if s.shot_number == args.shot
    ]
    if not wanted:
        nums = ", ".join(str(s.shot_number) for s in run.shots)
        print(f"no shot {args.shot}. This clip has: {nums}")
        return 2

    for s in wanted:
        print(report(s, has_height=args.height_cm is not None))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
