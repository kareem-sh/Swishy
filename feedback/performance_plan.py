"""Actionable performance plans and drill recommendations for reports."""

from typing import Dict, List, Tuple

from feedback.models import ShotSummary
from phase_detection.phases import PHASE_LABELS, PHASE_ORDER

# Drill paired to each biomechanical rule — one focused correction per rep.
RULE_DRILLS: Dict[str, str] = {
    "knee_flexion_loading": (
        "Load drill: 10 reps pausing at the bottom of your knee bend (hips back, chest tall) "
        "before rising into the shot."
    ),
    "hip_hinge_loading": (
        "Hip-hinge drill: practice 10 half-squats keeping shins vertical and hips sitting back "
        "without folding at the waist."
    ),
    "elbow_slot_ball_lift": (
        "One-hand form shooting from chest: 20 reps keeping elbow under the ball and pointing "
        "at the rim through the lift."
    ),
    "shoulder_alignment_lift": (
        "Wall touch drill: lift the ball while keeping shooting shoulder elevated — 15 slow reps "
        "without dropping the elbow."
    ),
    "trunk_posture": (
        "Tall-posture reps: shoot 10 times focusing on staying stacked (ears over hips) "
        "through the entire motion."
    ),
    "head_stability": (
        "Eyes-up drill: pick a rim spot and hold your gaze through release for 10 shots — "
        "no head dip or pull-away."
    ),
    "index_alignment_release": (
        "Finger-through drill: 20 close-range shots snapping index finger down through the ball "
        "at release."
    ),
    "elbow_extension_release": (
        "Extension pause: at the top of each shot, freeze with full elbow extension for 1 second "
        "before follow-through — 15 reps."
    ),
    "release_height": (
        "High-release drill: shoot from chin level only after the ball passes your forehead — "
        "10 slow reps emphasizing height."
    ),
    "follow_through_elbow": (
        "Hold finish: after every shot, keep elbow locked and arm extended for 2 seconds — 20 reps."
    ),
    "follow_through_index": (
        "Gooseneck hold: finish every rep with index finger pointing at the floor for 2 seconds — "
        "20 close-range reps."
    ),
    "landing_balance": (
        "Stick landing: after release, land in the same spot you jumped from — 10 jump shots "
        "marking your take-off point."
    ),
}

ACTIVE_SHOT_PHASES = frozenset(PHASE_ORDER) - {"ready_stance"}


def phases_before(phase: str) -> List[str]:
    if phase not in PHASE_ORDER:
        return []
    idx = PHASE_ORDER.index(phase)
    return PHASE_ORDER[:idx]


def phases_after(phase: str) -> List[str]:
    if phase not in PHASE_ORDER:
        return []
    idx = PHASE_ORDER.index(phase)
    return PHASE_ORDER[idx + 1 :]


def compute_missing_phases(
    phases_seen: List[str],
    entry_phase: str | None,
    started_mid_phase: bool,
    ended_early: bool,
) -> Tuple[List[str], List[str]]:
    """Return (missing_at_start, missing_at_end) phase keys."""
    missing_start: List[str] = []
    missing_end: List[str] = []

    if started_mid_phase and entry_phase:
        missing_start = [p for p in phases_before(entry_phase) if p not in phases_seen]

    if ended_early:
        last = phases_seen[-1] if phases_seen else None
        if last and last != "landing":
            missing_end = [p for p in phases_after(last) if p not in phases_seen]

    return missing_start, missing_end


def build_capture_note(
    started_mid_phase: bool,
    ended_early: bool,
    entry_phase: str | None,
    missing_start: List[str],
    missing_end: List[str],
) -> str:
    parts: List[str] = []

    if started_mid_phase and entry_phase:
        label = PHASE_LABELS.get(entry_phase, entry_phase)
        parts.append(
            f"Recording started mid-shot at {label}. "
            f"Earlier phases were not captured."
        )
        if missing_start:
            names = ", ".join(PHASE_LABELS.get(p, p) for p in missing_start)
            parts.append(f"Missing from start: {names}.")

    if ended_early:
        parts.append(
            "Shot ended before a full landing was detected (video cut off or session stopped). "
            "Release and follow-through were still analyzed where visible."
        )
        if missing_end:
            names = ", ".join(PHASE_LABELS.get(p, p) for p in missing_end)
            parts.append(f"Missing at end: {names}.")

    if not parts:
        return "Full shot cycle captured from load through landing."

    return " ".join(parts)


def build_shot_performance_plan(summary: ShotSummary) -> ShotSummary:
    """Attach drills, focus items, and capture notes to a scored shot."""
    missing_start, missing_end = compute_missing_phases(
        summary.phases_seen,
        summary.entry_phase,
        summary.started_mid_phase,
        summary.ended_early,
    )
    summary.missing_phases = missing_start + missing_end
    summary.capture_note = build_capture_note(
        summary.started_mid_phase,
        summary.ended_early,
        summary.entry_phase,
        missing_start,
        missing_end,
    )

    drills: List[str] = []
    focus: List[str] = []
    actions: List[str] = []

    sorted_violations = sorted(
        summary.violations,
        key=lambda r: {"error": 3, "warning": 2, "info": 1}.get(r.severity, 0),
        reverse=True,
    )

    for violation in sorted_violations[:3]:
        focus.append(violation.message)
        drill = RULE_DRILLS.get(violation.rule_id)
        if drill and drill not in drills:
            drills.append(drill)
        actions.append(
            f"Fix {violation.name}: {violation.message}"
        )

    if summary.started_mid_phase:
        actions.insert(
            0,
            "Next session: start recording before you begin the load so knee bend and ball lift can be scored.",
        )

    if summary.ended_early:
        actions.append(
            "Let the rep finish through landing before stopping the camera — landing balance is part of your score.",
        )

    if not focus and summary.score >= 75:
        focus.append("Maintain current mechanics — add game-speed reps and track consistency.")
        drills.append(
            "Game-speed drill: 5 makes from 5 spots at game pace, one form check between each spot."
        )
    elif not focus:
        focus.append("Break the shot into phases: load → lift → release → hold finish.")

    summary.next_rep_focus = focus[:2]
    summary.practice_drills = drills[:3]
    summary.performance_actions = actions[:4]
    return summary
