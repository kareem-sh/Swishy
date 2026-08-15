"""
Shot types and the registry that says which ones Swichy can actually analyse.

Two different negatives must never be conflated:

    "this is not a basketball shot"          -> RejectionReason.*
    "this IS a shot we recognise, but no
     analyser exists for it yet"             -> SUPPORT.NOT_IMPLEMENTED

The second is a roadmap statement, not a verdict about the video. A layup is
a real basketball shot; we simply have not built its biomechanics yet, and
scoring it with jump-shot rules would be wrong rather than merely imprecise.

Adding a shot type later means: research the literature, derive phases and
kinematic variables from it, check they are observable from MediaPipe
landmarks and the camera view, and only then register an analyser. See
docs/SHOT_TYPE_RESEARCH.md. Do not start from thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ShotType(str, Enum):
    JUMP_SHOT = "jump_shot"
    SET_SHOT = "set_shot"          # includes the free throw
    LAYUP = "layup"
    HOOK_SHOT = "hook_shot"
    DUNK = "dunk"
    UNKNOWN = "unknown"


class Support(str, Enum):
    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"


class RejectionReason(str, Enum):
    """Why a video produced no analysis. None of these is a score of zero."""

    INVALID_VIDEO = "invalid_video"              # unreadable / bad metadata
    NO_PERSON_DETECTED = "no_person_detected"    # e.g. pizza.mp4
    NO_SHOOTING_EVENT = "no_shooting_event"      # a person, but nobody shot
    SHOT_TYPE_NOT_SUPPORTED_YET = "shot_type_not_supported_yet"


@dataclass(frozen=True)
class ShotTypeInfo:
    shot_type: ShotType
    support: Support
    label: str
    note: str = ""

    @property
    def is_implemented(self) -> bool:
        return self.support is Support.IMPLEMENTED


# The registry. Adding a production analyser means flipping one entry to
# IMPLEMENTED *after* its phases and rules have been derived from evidence.
SHOT_TYPE_REGISTRY: dict[ShotType, ShotTypeInfo] = {
    ShotType.JUMP_SHOT: ShotTypeInfo(
        ShotType.JUMP_SHOT,
        Support.IMPLEMENTED,
        "Jump Shot",
    ),
    ShotType.SET_SHOT: ShotTypeInfo(
        ShotType.SET_SHOT,
        Support.IMPLEMENTED,
        "Set Shot / Free Throw",
    ),
    ShotType.LAYUP: ShotTypeInfo(
        ShotType.LAYUP,
        Support.NOT_IMPLEMENTED,
        "Layup",
        "Driving approach and a one-footed take-off. Different phases and "
        "different joint demands from a jump shot; needs its own model.",
    ),
    ShotType.HOOK_SHOT: ShotTypeInfo(
        ShotType.HOOK_SHOT,
        Support.NOT_IMPLEMENTED,
        "Hook Shot",
        "Sideways-facing release with an extended arm arc. The shooting arm "
        "leaves the sagittal plane, so sagittal angle rules do not transfer.",
    ),
    ShotType.DUNK: ShotTypeInfo(
        ShotType.DUNK,
        Support.NOT_IMPLEMENTED,
        "Dunk",
        "No ball release in the projectile sense; the ball is carried to the "
        "rim. Release-based rules are meaningless here.",
    ),
    ShotType.UNKNOWN: ShotTypeInfo(
        ShotType.UNKNOWN,
        Support.NOT_IMPLEMENTED,
        "Unrecognised action",
    ),
}


def describe(shot_type: ShotType) -> ShotTypeInfo:
    return SHOT_TYPE_REGISTRY.get(shot_type, SHOT_TYPE_REGISTRY[ShotType.UNKNOWN])


def is_implemented(shot_type: ShotType) -> bool:
    return describe(shot_type).is_implemented


@dataclass(frozen=True)
class ShotClassification:
    """The classifier's verdict for one candidate attempt."""

    shot_type: ShotType
    confidence: float
    evidence: tuple[str, ...] = ()
    rejection: Optional[RejectionReason] = None

    @property
    def is_implemented(self) -> bool:
        return is_implemented(self.shot_type)

    @property
    def label(self) -> str:
        return describe(self.shot_type).label
