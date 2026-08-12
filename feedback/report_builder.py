"""Build detailed shot and session reports from captured data."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from feedback.performance_plan import build_shot_performance_plan
from feedback.models import ShotSummary
from feedback.report_models import (
    DetailedShotReport,
    KeyFrame,
    PhaseMoment,
    RuleEvaluation,
    SessionReport,
    VisibilityGapNote,
)
from phase_detection.phases import PHASE_LABELS
from utils.config_loader import load_yaml


def build_detailed_shot_report(
    summary: ShotSummary,
    phase_moments: List[dict],
    key_frame_pairs: List[tuple],
    start_ms: int,
    end_ms: int,
    visibility_gaps: Optional[List[VisibilityGapNote]] = None,
) -> DetailedShotReport:
    biomech = load_yaml("biomechanics.yaml")
    rules_cfg = biomech.get("rules", {})

    rule_map = {r.rule_id: r for r in summary.passed_rules + summary.violations}
    evaluations: List[RuleEvaluation] = []

    for rule_id, rule in rule_map.items():
        cfg = rules_cfg.get(rule_id, {})
        evaluations.append(
            RuleEvaluation(
                rule_id=rule_id,
                name=rule.name,
                passed=rule.passed,
                severity=rule.severity,
                message=rule.message,
                rationale=cfg.get("rationale", ""),
                phase=rule.phase,
                measured_value=rule.measured_value,
                min_value=rule.min_value,
                max_value=rule.max_value,
            )
        )

    evaluations.sort(key=lambda r: (r.passed, r.severity != "error", r.severity != "warning"))

    timeline = [
        PhaseMoment(
            phase=m["phase"],
            phase_label=m.get("phase_label") or PHASE_LABELS.get(m["phase"], m["phase"]),
            timestamp_ms=m["timestamp_ms"],
            timestamp_label=_ts_label(m["timestamp_ms"]),
            frame_index=m["frame_index"],
        )
        for m in phase_moments
    ]

    shot_prefix = f"shot_{summary.shot_number:02d}"
    populated_frames: List[KeyFrame] = []
    frame_images: Dict[int, np.ndarray] = {}

    for kf, img in key_frame_pairs:
        filename = f"{shot_prefix}_frame_{kf.frame_index:05d}_{kf.phase}.jpg"
        kf.image_filename = filename
        populated_frames.append(kf)
        frame_images[kf.frame_index] = img

    violation_frames = {rid: kf.image_filename for kf in populated_frames for rid in kf.rule_ids}
    for ev in evaluations:
        if ev.rule_id in violation_frames:
            ev.key_frame_filename = violation_frames[ev.rule_id]

    summary = build_shot_performance_plan(summary)

    return DetailedShotReport(
        summary=summary,
        phase_timeline=timeline,
        rule_evaluations=evaluations,
        key_frames=populated_frames,
        frame_images=frame_images,
        visibility_gaps=visibility_gaps or [],
        start_timestamp_ms=start_ms,
        end_timestamp_ms=end_ms,
    )


def build_session_report(
    session_id: str,
    source_type: str,
    source_name: str,
    fps: float,
    total_frames: int,
    shots: List[DetailedShotReport],
) -> SessionReport:
    report = SessionReport(
        session_id=session_id,
        source_type=source_type,
        source_name=source_name,
        fps=fps,
        total_frames=total_frames,
        shots=shots,
    )

    if not shots:
        report.overall_grade = "N/A"
        report.top_improvements = [
            "No complete shots detected. Ensure full body is visible and perform a full jump shot."
        ]
        return report

    scored_shots = [s for s in shots if s.summary.score is not None]
    scores = [int(s.summary.score) for s in scored_shots]
    if scores:
        report.overall_score = round(sum(scores) / len(scores))
        report.overall_grade = _grade(report.overall_score)
    else:
        report.overall_score = None
        report.overall_grade = "N/A"

    violation_counts: Counter = Counter()
    strength_counts: Counter = Counter()
    for shot in shots:
        for ev in shot.rule_evaluations:
            if ev.passed:
                strength_counts[ev.name] += 1
            else:
                violation_counts[ev.name] += 1

    report.top_improvements = [
        f"{name} (failed in {count}/{len(scored_shots)} scored shots)"
        for name, count in violation_counts.most_common(5)
    ]
    report.strengths = [
        f"{name} (passed in {count}/{len(scored_shots)} scored shots)"
        for name, count in strength_counts.most_common(5)
    ]

    report.session_notes = _session_capture_notes(shots)
    report.practice_plan = _build_session_practice_plan(
        scored_shots, violation_counts
    )

    return report


def _session_capture_notes(shots: List[DetailedShotReport]) -> List[str]:
    notes: List[str] = []
    mid_starts = [s for s in shots if s.summary.started_mid_phase]
    early_ends = [s for s in shots if s.summary.ended_early]

    if mid_starts:
        nums = ", ".join(str(s.summary.shot_number) for s in mid_starts)
        notes.append(
            f"Shot(s) {nums} started mid-rep — camera was on after the load began. "
            f"Start recording before you dip to score leg drive and ball lift."
        )
    if early_ends:
        nums = ", ".join(str(s.summary.shot_number) for s in early_ends)
        notes.append(
            f"Shot(s) {nums} ended before landing — analysis still covers visible phases "
            f"but landing balance was not scored."
        )
    return notes


def _build_session_practice_plan(
    shots: List[DetailedShotReport],
    violation_counts: Counter,
) -> List[str]:
    plan: List[str] = []
    seen_drills: set[str] = set()

    for shot in shots:
        for drill in shot.summary.practice_drills:
            if drill not in seen_drills:
                plan.append(drill)
                seen_drills.add(drill)

    if violation_counts:
        top_name, top_count = violation_counts.most_common(1)[0]
        plan.insert(
            0,
            f"Session priority: fix '{top_name}' — it failed in {top_count}/{len(shots)} shot(s). "
            f"Do one drill below before your next live session.",
        )
    elif shots:
        plan.append(
            "Strong session overall — add game-speed reps and film yourself from the side "
            "to keep mechanics under pressure."
        )

    return plan[:6]


def _grade(score: int) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Needs Work"


def _ts_label(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"
