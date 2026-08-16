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
    "guide_hand_support_lift": (
        "Guide-hand form shooting: take 15 close-range reps with the guide hand supporting "
        "the side during the lift, then coming cleanly off the ball at release."
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
    "jump_release_timing": (
        "Peak-release drill: take 10 controlled jump shots and let the ball leave on the rise "
        "or at the top, before your body starts coming down."
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

# REMOVED: ACTIVE_SHOT_PHASES.
# It told the tracker which phases meant "a shot is underway". The tracker now
# asks that question of the DETECTOR's four states (phase_detection.phases
# .CORE_ACTIVE), not of the coaching vocabulary, because deciding whether a
# shot is happening and deciding what to call the part of it you are in are
# different jobs. Nothing else referenced it.


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


def _lifted_off_on_a_set_shot(summary: ShotSummary) -> bool:
    """Was this a set shot whose feet still left the floor?

    Read from the classifier's own notes rather than re-deriving the
    elevation. The classifier already measured it, already applied both
    thresholds, and already declined to call it a jump shot -- recomputing here
    would mean two places that could disagree about the same attempt.
    """
    classification = getattr(summary, "classification", None)
    if classification is None:
        return False
    if getattr(summary.shot_type, "name", "") != "SET_SHOT":
        return False
    # The classifier calls this field `evidence`: the tuple of measured
    # statements behind its verdict.
    return any("still left the floor" in note
               for note in (getattr(classification, "evidence", ()) or ()))


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

    if summary.score is None:
        # Unsupported/rejected attempts deliberately have no biomechanics
        # score. Do not compare that missing score with numeric thresholds or
        # generate jump/set-shot coaching for an unsupported shot type.
        summary.next_rep_focus = []
        summary.practice_drills = []
        return summary

    drills: List[str] = []
    focus: List[str] = []

    sorted_violations = sorted(
        summary.violations,
        key=lambda r: {"error": 3, "warning": 2, "info": 1}.get(r.severity, 0),
        reverse=True,
    )

    # A set shot the player did not stay down for.
    #
    # This is not a rule violation in `biomechanics.yaml`, because it is not a
    # property of any frame or phase -- it is a property of the WHOLE attempt,
    # decided by the classifier from the elevation it measured. The rule engine
    # has no way to express "the shot as a whole was between two categories".
    #
    # It goes first because it is more actionable than any joint angle: the
    # player is doing something they can change by deciding to, and the
    # half-committed version is the least repeatable of the three options.
    if _lifted_off_on_a_set_shot(summary):
        focus.append(
            "Your feet left the floor, but not enough to be a jump shot. "
            "Pick one and repeat it: stay planted through the release, or "
            "commit to the jump. Half of each is the hardest to repeat. On a "
            "free throw, staying planted is also the safer choice — leaving "
            "the floor is legal, but landing over the line before the ball "
            "reaches the ring is a violation."
        )

    seen = set()
    for violation in sorted_violations[:3]:
        if violation.message in seen:
            continue
        seen.add(violation.message)
        focus.append(violation.message)
        drill = RULE_DRILLS.get(violation.rule_id)
        if drill and drill not in drills:
            drills.append(drill)

    if not focus and summary.score >= 75:
        focus.append(
            "Nothing to change. Keep these reps and start shooting them at game speed."
        )
        drills.append(
            "Game speed: 5 makes from 5 spots, one form check between each spot."
        )
    elif not focus:
        focus.append("Take it one piece at a time: load, lift, release, hold the finish.")

    summary.next_rep_focus = focus[:2]
    summary.practice_drills = drills[:3]
    return summary
