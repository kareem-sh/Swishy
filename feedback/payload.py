"""
Build the JSON payload for one analysed video.

THE RULE THIS FILE EXISTS TO ENFORCE
------------------------------------
Everything here is measured. Nothing is invented, and nothing is defaulted to
a plausible-looking constant.

The serialiser this replaces did the opposite. It took these arguments:

    shot_type       = "jump_shot"                        # never read the classifier
    court_location  = "right_wing_three_point_line"      # nothing detects court position
    shot_start_time_ms = 1000, shot_end_time_ms = 2500   # placeholders
    is_placeholder  = True

so every shot was reported as a jump shot from the right wing, whatever had
actually happened, while the classifier's real verdict was computed and thrown
away. `court_location` is gone rather than nulled: Swichy has no court
detection, so the field has no meaning. A missing field says "we do not
measure this". A fabricated one says something false.

At the same time the payload OMITTED the product's headline feature. Per-phase
scores, the rejection reason, and the classifier's evidence were all absent.
They are all here now.
"""

from __future__ import annotations

from typing import List, Optional

from feedback.localization import (
    localized_coaching,
    phase_label_ar,
    rule_feedback_ar,
    rule_name_ar,
)
from feedback.models import PhaseScore, ShotSummary
from shots.types import describe


def _rule_dict(rule) -> dict:
    """A rule result plus the measurement behind it.

    Every coaching claim ships with the number that produced it. That is what
    separates this from a template: "your elbow is low" is an opinion,
    "your elbow is low — measured 118 degrees, target 145-180" is a reading.
    """
    # A rule's own 0-100 score is the exact credit used by the phase scorer
    # before severity weighting. Measured-only rules deliberately expose null:
    # displaying a number is not the same as grading it.
    rule_score = int(round(100.0 * rule.credit)) if rule.scored else None
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "name_localized": {
            "en": rule.name,
            "ar": rule_name_ar(rule.rule_id, rule.name),
        },
        "phase": rule.phase,
        "score": rule_score,
        "passed": rule.passed,
        "outcome": rule.outcome.value,
        "severity": rule.severity,
        # Keep `message` untouched for existing frontend consumers. New code
        # can select a stable language key from `feedback`.
        "message": rule.message,
        "feedback": {
            "en": rule.message,
            "ar": rule_feedback_ar(rule),
        },
        "measured_value": rule.measured_value,
        "unit": rule.unit,
        "target_min": rule.min_value,
        "target_max": rule.max_value,
        "ideal_min": rule.ideal_min,
        "ideal_max": rule.ideal_max,
        "scored": rule.scored,
    }


def _phase_dict(phase: PhaseScore) -> dict:
    return {
        "phase": phase.phase,
        "label": phase.label,
        "label_localized": {
            "en": phase.label,
            "ar": phase_label_ar(phase.phase, phase.label),
        },
        "score": phase.score,
        "grade": phase.grade,
        "on_target": list(phase.strengths),
        "refine": list(phase.refinements),
        "change": list(phase.fixes),
        "rules": [_rule_dict(r) for r in phase.rules],
        "measured_only": [_rule_dict(r) for r in phase.measured],
    }


def _coaching_dict(summary: ShotSummary) -> dict:
    """Existing English coaching arrays plus a bilingual additive view."""
    summary_items = list(summary.coaching_tips)
    focus_items = list(summary.next_rep_focus)
    drill_items = list(summary.practice_drills)
    rules = [
        rule
        for phase in summary.phase_scores
        for rule in (phase.rules + phase.measured)
    ]
    # A synthetic/API caller may populate passed_rules or violations without
    # constructing phase scores. Include those as a translation fallback.
    rules.extend(summary.passed_rules)
    rules.extend(summary.violations)

    return {
        # Unchanged legacy fields.
        "summary": summary_items,
        "focus_next_rep": focus_items,
        "drills": drill_items,
        # Additive bilingual fields for new clients.
        "localized": {
            "summary": localized_coaching(summary_items, rules),
            "focus_next_rep": localized_coaching(focus_items, rules),
            "drills": localized_coaching(drill_items, rules),
        },
    }


def _shot_type_dict(summary: ShotSummary) -> dict:
    """The classifier's real verdict, with its confidence and its reasoning."""
    if summary.shot_type is None:
        return {"id": None, "label": None, "supported": False,
                "confidence": None, "evidence": []}
    info = describe(summary.shot_type)
    classification = summary.classification
    return {
        "id": summary.shot_type.value,
        "label": info.label,
        "supported": info.is_implemented,
        "note": info.note or None,
        "confidence": (
            round(classification.confidence, 3) if classification else None
        ),
        # e.g. "feet rose 0.207 body-heights at the shooting event (>= 0.12)"
        "evidence": list(classification.evidence) if classification else [],
    }


def _outcome_dict(outcome) -> Optional[dict]:
    """Ball outcome, or None when the ball was not tracked.

    None means "not measured". It is never flattened into a made/missed guess.
    """
    if outcome is None:
        return None
    return {
        "result": outcome.result,
        # Derived from `result`, kept so a client does not have to know the
        # vocabulary. This is a convenience field, not a second opinion.
        "is_basket": outcome.result == "made",
        "confidence": outcome.confidence,
        "release_frame": outcome.release_frame,
        "release_timestamp_ms": outcome.release_timestamp_ms,
        "entry_frame": outcome.entry_frame,
        "entry_timestamp_ms": outcome.entry_timestamp_ms,
        "outcome_timestamp_ms": outcome.outcome_timestamp_ms,
        "evidence": list(outcome.evidence),
        "release_timing": outcome.timeseries_summary.get("release_timing"),
        "trajectory_comparison": outcome.timeseries_summary.get(
            "trajectory_comparison"
        ),
    }


def shot_to_dict(
    summary: ShotSummary,
    start_timestamp_ms: Optional[int] = None,
    end_timestamp_ms: Optional[int] = None,
) -> dict:
    """One shot as JSON data.

    Timestamps are Optional and default to None rather than to a placeholder:
    a caller that does not know when the shot began says so.
    """
    start = start_timestamp_ms if start_timestamp_ms is not None else summary.start_timestamp_ms
    end = end_timestamp_ms if end_timestamp_ms is not None else summary.end_timestamp_ms
    payload = {
        "shot_number": summary.shot_number,
        "shot_type": _shot_type_dict(summary),
        "scored": not summary.is_rejected,
        "start_timestamp_ms": start,
        "end_timestamp_ms": end,
        "duration_s": round((end - start) / 1000.0, 2)
        if start is not None and end is not None else None,
        "release_timing": {
            "pose_release_timestamp_ms": summary.pose_release_timestamp_ms,
            "ball_release_timestamp_ms": summary.ball_release_timestamp_ms,
            "release_disagreement_ms": summary.release_disagreement_ms,
            "ball_minus_pose_release_ms": summary.ball_minus_pose_release_ms,
            "release_disagreement_limit_ms": (
                summary.release_disagreement_limit_ms
            ),
            "release_timing_status": summary.release_timing_status,
            "release_alignment_confidence": (
                summary.release_alignment_confidence
            ),
        },
        "outcome": _outcome_dict(summary.outcome),
    }

    if summary.is_rejected:
        # A rejection is not a score of zero. No score, no phase scores, no
        # coaching -- because no analyser ran on this attempt.
        payload.update({
            "score": None,
            "grade": None,
            "rejection": {
                "reason": summary.rejection.value if summary.rejection else None,
                "explanation": _rejection_text(summary),
            },
            "phases": [],
            "coaching": {},
        })
        return payload

    payload.update({
        "score": summary.score,
        "grade": summary.grade,
        "rejection": None,
        "rules": {
            "passed": summary.passed_count,
            "evaluated": summary.total_count,
        },
        "phases": [_phase_dict(p) for p in summary.phase_scores],
        "coaching": _coaching_dict(summary),
        "capture": {
            "note": summary.capture_note or None,
            "started_mid_shot": summary.started_mid_phase,
            "ended_early": summary.ended_early,
            "phases_seen": list(summary.phases_seen),
        },
    })
    return payload


def _rejection_text(summary: ShotSummary) -> str:
    info = describe(summary.shot_type) if summary.shot_type else None
    if info is not None and not info.is_implemented:
        return (
            f"This looks like a {info.label.lower()}. Swichy does not analyse "
            "that shot yet, so it has not been scored."
        )
    return "This attempt could not be analysed, so it has not been scored."


def session_to_dict(
    video_name: str,
    fps: float,
    frames_read: int,
    shots: List[ShotSummary],
    discarded_candidates: int = 0,
    rejection: Optional[str] = None,
    rejection_detail: str = "",
) -> dict:
    """The whole analysis of one uploaded video."""
    scored = [s for s in shots if not s.is_rejected]
    average = (
        round(sum(s.score for s in scored) / len(scored)) if scored else None
    )
    return {
        "video": video_name,
        "fps": round(fps, 3),
        "frames_analysed": frames_read,
        "analysed": rejection is None,
        "rejection": (
            {"reason": rejection, "explanation": rejection_detail}
            if rejection else None
        ),
        "shot_count": len(shots),
        "scored_shot_count": len(scored),
        "discarded_candidates": discarded_candidates,
        "average_score": average,
        "shots": [shot_to_dict(s) for s in shots],
    }
