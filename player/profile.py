"""
Player context that is stable across a session, kept separate from per-frame
pose data.

Height policy
-------------
Height is **optional and user-provided**. It is never estimated from the
camera, and never defaulted to a population average.

A single camera cannot recover absolute scale: projecting a 3D world onto a 2D
image leaves size and distance ambiguous. MediaPipe's world landmarks come from
fitting the GHUM statistical body model, so they carry anatomical priors and
regress toward population means rather than measuring *this* player. The best
published MediaPipe height result (166 subjects, 1.54 +/- 0.64% error) required
a physical reference object in frame, which Swichy does not have.

So `HeightSource.VISION` exists in the enum as a reserved value, but nothing
produces it. If vision estimation is added later it must be validated against
known-height players before it is allowed anywhere near scoring, and it must
never silently override a value the player typed in.

When height is unknown, `height_cm` stays None and every height-dependent
metric is skipped. Height-independent metrics (all joint angles, trunk lean)
are unaffected -- the system stays fully usable without it.

See resources.md Part B for the evidence behind this policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from utils.config_loader import load_yaml

# Plausibility bounds. Deliberately wide: this is a typo guard (someone entering
# inches, or slipping a digit), not a claim about who may use the app.
MIN_PLAUSIBLE_HEIGHT_CM = 120.0
MAX_PLAUSIBLE_HEIGHT_CM = 230.0


class HeightSource(str, Enum):
    """Provenance of the height value. Not a confidence score."""

    USER = "user"
    VISION = "vision"  # RESERVED - never produced; see module docstring
    NONE = "none"


@dataclass(frozen=True)
class PlayerProfile:
    """Session-level context for the player being coached."""

    height_cm: Optional[float] = None
    height_source: HeightSource = HeightSource.NONE
    shooting_hand: str = "auto"
    label: str = ""

    @property
    def has_height(self) -> bool:
        return self.height_cm is not None

    @property
    def height_m(self) -> Optional[float]:
        return None if self.height_cm is None else self.height_cm / 100.0

    def normalized(self, metres: Optional[float]) -> Optional[float]:
        """Express a vertical measurement as a fraction of the player's height.

        Returns None when either the measurement or the height is unavailable,
        so callers degrade to "not measured" rather than to a wrong number.
        """
        if metres is None or self.height_m is None or self.height_m <= 0:
            return None
        return metres / self.height_m

    def describe_height(self) -> str:
        if not self.has_height:
            return "not provided"
        return f"{self.height_cm:.0f} cm ({self.height_source.value})"


def _coerce_height(raw) -> tuple[Optional[float], Optional[str]]:
    """Validate a user-entered height. Returns (height_cm, warning)."""
    if raw is None or raw == "":
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, f"Ignoring player height {raw!r}: not a number."

    if value <= 0:
        return None, f"Ignoring player height {value}: must be positive."

    if not (MIN_PLAUSIBLE_HEIGHT_CM <= value <= MAX_PLAUSIBLE_HEIGHT_CM):
        return None, (
            f"Ignoring player height {value:g} cm: outside the plausible range "
            f"{MIN_PLAUSIBLE_HEIGHT_CM:g}-{MAX_PLAUSIBLE_HEIGHT_CM:g} cm. "
            "Height must be in centimetres (e.g. 180), not inches or metres."
        )

    return value, None


def build_player_profile(
    height_cm=None,
    shooting_hand: str = "auto",
    label: str = "",
    warn=print,
) -> PlayerProfile:
    """Build a validated profile from raw values, rejecting implausible input."""
    height, warning = _coerce_height(height_cm)
    if warning and warn is not None:
        warn(f"Warning: {warning}")

    hand = str(shooting_hand or "auto").strip().lower()
    if hand not in {"auto", "left", "right"}:
        if warn is not None:
            warn(f"Warning: unknown shooting_hand {shooting_hand!r}; using auto")
        hand = "auto"

    return PlayerProfile(
        height_cm=height,
        height_source=HeightSource.USER if height is not None else HeightSource.NONE,
        shooting_hand=hand,
        label=str(label or ""),
    )


def load_player_profile(warn=print) -> PlayerProfile:
    """Load the player profile from config/player.yaml.

    A missing or empty file is a valid state meaning "no height provided".
    """
    try:
        cfg = load_yaml("player.yaml") or {}
    except FileNotFoundError:
        cfg = {}

    return build_player_profile(
        height_cm=cfg.get("height_cm"),
        shooting_hand=cfg.get("shooting_hand", "auto"),
        label=cfg.get("label", ""),
        warn=warn,
    )
