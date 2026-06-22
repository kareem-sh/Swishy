"""Build detailed shot and session reports from captured data."""

from collections import Counter
from typing import Dict, List

import numpy as np

from feedback.models import ShotSummary
from feedback.report_models import (
    DetailedShotReport,
    KeyFrame,
    PhaseMoment,
    RuleEvaluation,
    SessionReport,
)
from phase_detection.phases import PHASE_LABELS
from utils.config_loader import load_yaml


def build_detailed_shot_report(
    summary: ShotSummary,
    phase_moments: List[dict],
    key_frame_pairs: List[tuple],
    start_ms: int,
    end_ms: int,
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

    return DetailedShotReport(
        summary=summary,
        phase_timeline=timeline,
        rule_evaluations=evaluations,
        key_frames=populated_frames,
        frame_images=frame_images,
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

    scores = [s.summary.score for s in shots if s.summary.total_count > 0]
    report.overall_score = round(sum(scores) / len(scores)) if scores else 0
    report.overall_grade = _grade(report.overall_score)

    violation_counts: Counter = Counter()
    strength_counts: Counter = Counter()
    for shot in shots:
        for ev in shot.rule_evaluations:
            if ev.passed:
                strength_counts[ev.name] += 1
            else:
                violation_counts[ev.name] += 1

    report.top_improvements = [
        f"{name} (failed in {count}/{len(shots)} shots)"
        for name, count in violation_counts.most_common(5)
    ]
    report.strengths = [
        f"{name} (passed in {count}/{len(shots)} shots)"
        for name, count in strength_counts.most_common(5)
    ]

    return report


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
