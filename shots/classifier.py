"""
Classify a completed shot candidate into a shot type.

WHAT THIS CAN AND CANNOT DO
---------------------------
From pose landmarks alone, in a practice setting, two discriminators are
reliable:

  1. VERTICAL displacement of the feet during the attempt
        -> separates a jump shot from a set shot / free throw
  2. HORIZONTAL travel of the hips during the attempt
        -> separates a stationary shot from a driving action

That is enough for the current product scope, which is stationary shooting.
It is NOT enough to tell a layup from a hook shot from a dunk. Those share
"the player is moving", and separating them needs approach direction,
take-off foot, and ball position -- none of which we model yet.

So a driving action is reported as LAYUP, the most common member of that
family, with LOW confidence and an explicit note. It is never scored: every
member of that family is NOT_IMPLEMENTED, so the outcome (a rejection with
no score) is identical regardless of which one it actually was. The label is
a hint for the user, not a claim.

WHY NOT "jump detected = jump shot"
-----------------------------------
A player bouncing on their toes while settling produces small vertical
motion that would trip a naive jump test. The classifier therefore measures
vertical displacement AT THE SHOOTING EVENT -- around ball lift through
release -- not the maximum over the whole clip.
"""

from __future__ import annotations

from dataclasses import dataclass

from shots.types import RejectionReason, ShotClassification, ShotType

# --------------------------------------------------------------------------
# Thresholds — EMPIRICAL / PROJECT-CALIBRATED, not published norms.
#
# Both are expressed as fractions of the player's own on-screen height, which
# makes them invariant to camera distance and zoom. They are measured in IMAGE
# space, because MediaPipe world landmarks are hip-centred and therefore cannot
# observe whole-body translation at all.
#
# Vertical: the literature reports vertical displacement of 15.3 +/- 5.1 cm for
# free throws versus 26.9-31.2 cm for jump shots (resources.md A1). For a
# ~1.8 m player that is roughly 0.085 vs 0.15-0.17 body heights, so the classes
# separate near 0.12. That study measured posterior-calcaneus-to-ground while we
# measure ankle-landmark rise above a standing baseline, so the figures are
# related but not interchangeable. Calibrated, not derived; revisit against
# labelled data.
JUMP_VERTICAL_DISPLACEMENT_RATIO = 0.12

# Horizontal: fraction of frame width the hips traverse during the attempt. A
# stationary shooter sways within a narrow band; a driving player crosses the
# frame. Purely empirical.
DRIVING_HORIZONTAL_TRAVEL_RATIO = 0.18

# Backwards-compatible aliases (the units changed, the names are kept).
JUMP_VERTICAL_DISPLACEMENT_M = JUMP_VERTICAL_DISPLACEMENT_RATIO
DRIVING_HORIZONTAL_TRAVEL_M = DRIVING_HORIZONTAL_TRAVEL_RATIO


@dataclass
class AttemptEvidence:
    """Measurements taken over one candidate attempt, from ball lift to release."""

    vertical_displacement_m: float = 0.0
    # False when the stance before the attempt could not be observed, so the
    # elevation above is a fallback rather than a measurement. Defaults True
    # because evidence built by hand in tests supplies a real figure by
    # definition; only the pipeline can discover that it failed to measure.
    vertical_displacement_measured: bool = True
    horizontal_travel_m: float = 0.0
    wrist_rise_m: float = 0.0
    wrist_above_shoulder_m: float = -1.0
    reached_release: bool = False
    duration_s: float = 0.0


# Ceiling on confidence when elevation came from a whole-clip reference rather
# than the player's own stance. Deliberately below the 0.6 floor of a measured
# verdict, so a provisional type can never outrank a measured one.
UNMEASURED_CONFIDENCE_CEILING = 0.5


def _capped(confidence: float, evidence: "AttemptEvidence") -> float:
    if evidence.vertical_displacement_measured:
        return confidence
    return min(confidence, UNMEASURED_CONFIDENCE_CEILING)


def classify(evidence: AttemptEvidence) -> ShotClassification:
    """Decide which shot type an attempt represents.

    Decision order matters: "is the player driving?" is asked before "did they
    jump?", because a layup involves a jump too. Asking about the jump first
    would label every layup a jump shot.
    """
    notes: list[str] = []

    if not evidence.reached_release:
        return ShotClassification(
            shot_type=ShotType.UNKNOWN,
            confidence=0.0,
            evidence=("no release detected",),
            rejection=RejectionReason.NO_SHOOTING_EVENT,
        )

    # 1. Driving actions leave the stationary-shooting family entirely.
    if evidence.horizontal_travel_m >= DRIVING_HORIZONTAL_TRAVEL_RATIO:
        notes.append(
            f"hips travelled {evidence.horizontal_travel_m:.2f} of frame width during the "
            f"attempt (>= {DRIVING_HORIZONTAL_TRAVEL_RATIO:.2f}), so this is a "
            "moving action, not a stationary shot"
        )
        notes.append(
            "pose alone cannot separate layup / hook shot / dunk; reported as "
            "layup with low confidence"
        )
        return ShotClassification(
            shot_type=ShotType.LAYUP,
            confidence=0.35,
            evidence=tuple(notes),
            rejection=RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET,
        )

    # 2. Stationary. Vertical displacement at the shooting event separates
    #    a jump shot from a set shot.
    notes.append(
        f"hips stayed within {evidence.horizontal_travel_m:.2f} of frame width — stationary"
    )

    # Elevation is measured against the player's own stance just before the
    # shot, which is the only reference that cancels perspective. When that
    # stance was never recorded -- a clip cut at the shot, or a camera started
    # late -- a whole-clip reference is used instead. That still answers the
    # question, just less reliably, so the answer is reported with its
    # confidence capped rather than withheld: refusing here would discard
    # every single-shot clip, which the whole-clip reference handles correctly.
    if not evidence.vertical_displacement_measured:
        notes.append(
            "no stance was recorded before this shot, so the feet are measured "
            "against the whole clip rather than the player's own starting "
            "position -- treat the type as provisional"
        )

    if evidence.vertical_displacement_m >= JUMP_VERTICAL_DISPLACEMENT_RATIO:
        notes.append(
            f"feet rose {evidence.vertical_displacement_m:.3f} body-heights at the shooting "
            f"event (>= {JUMP_VERTICAL_DISPLACEMENT_RATIO:.2f})"
        )
        margin = evidence.vertical_displacement_m - JUMP_VERTICAL_DISPLACEMENT_RATIO
        confidence = min(0.95, 0.6 + margin * 2.0)
        return ShotClassification(
            ShotType.JUMP_SHOT, _capped(confidence, evidence), tuple(notes)
        )

    notes.append(
        f"feet rose only {evidence.vertical_displacement_m:.3f} body-heights at the shooting "
        f"event (< {JUMP_VERTICAL_DISPLACEMENT_RATIO:.2f})"
    )
    margin = JUMP_VERTICAL_DISPLACEMENT_RATIO - evidence.vertical_displacement_m
    confidence = min(0.95, 0.6 + margin * 3.0)
    return ShotClassification(
        ShotType.SET_SHOT, _capped(confidence, evidence), tuple(notes)
    )
