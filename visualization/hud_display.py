"""Temporal smoothing for on-screen HUD so values stay readable during fast motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from angles.calculator import AngleResult
from utils.config_loader import load_yaml

if TYPE_CHECKING:
    from pipeline import FrameResult


@dataclass
class HudAngleLine:
    text: str
    is_valid: bool
    is_stable: bool


@dataclass
class HudViolationLine:
    message: str
    severity: str


@dataclass
class HudDisplay:
    """Smoothed values for overlay rendering only (analysis uses raw data)."""

    phase_label: str = ""
    angles: Dict[str, HudAngleLine] = field(default_factory=dict)
    rules_line: str = ""
    violations: List[HudViolationLine] = field(default_factory=list)


class HudDisplaySmoother:
    """
    Debounces phase labels, smooths angle numbers, and holds coaching messages
    so the overlay does not flicker frame-to-frame during a fast jump shot.
    """

    _ANGLE_KEYS = ("elbow", "knee", "trunk")

    def __init__(self):
        cfg = load_yaml("display.yaml")
        self._phase_stable_frames = int(cfg.get("phase_stable_frames", 12))
        self._phase_min_hold_frames = int(cfg.get("phase_min_hold_frames", 24))
        self._angle_alpha = float(cfg.get("angle_ema_alpha", 0.08))
        self._angle_step = float(cfg.get("angle_step_degrees", 4))
        self._angle_min_hold = int(cfg.get("angle_min_hold_frames", 12))
        self._angle_invalid_hold = int(cfg.get("angle_invalid_hold_frames", 18))
        self._violation_hold = int(cfg.get("violation_hold_frames", 45))
        self._violation_max = int(cfg.get("violation_max_items", 3))

        self._display_phase_label = "Ready Stance"
        self._phase_candidate = "ready_stance"
        self._phase_candidate_label = "Ready Stance"
        self._phase_candidate_count = 0
        self._phase_hold_remaining = 0

        self._angle_ema: Dict[str, float] = {}
        self._angle_shown_int: Dict[str, int] = {}
        self._angle_hold_counter: Dict[str, int] = {}
        self._angle_invalid_counter: Dict[str, int] = {}
        self._angle_last_valid: Dict[str, HudAngleLine] = {}

        self._rules_line = ""
        self._rules_hold = 0
        self._violations: List[HudViolationLine] = []
        self._violation_hold_remaining = 0

    def update(self, frame_result: FrameResult) -> HudDisplay:
        if not frame_result.has_pose:
            return HudDisplay(phase_label=self._display_phase_label)

        self._update_phase(frame_result)
        angles = self._update_angles(frame_result)
        self._update_rules_and_violations(frame_result)

        return HudDisplay(
            phase_label=self._display_phase_label,
            angles=angles,
            rules_line=self._rules_line,
            violations=list(self._violations),
        )

    def reset(self):
        self.__init__()

    def _update_phase(self, frame_result: FrameResult):
        phase = frame_result.phase
        label = frame_result.phase_label or phase

        if phase == self._phase_candidate:
            self._phase_candidate_count += 1
        else:
            self._phase_candidate = phase
            self._phase_candidate_label = label
            self._phase_candidate_count = 1

        if self._phase_hold_remaining > 0:
            self._phase_hold_remaining -= 1
            return

        if self._phase_candidate_count >= self._phase_stable_frames:
            if label != self._display_phase_label:
                self._display_phase_label = label
                self._phase_hold_remaining = self._phase_min_hold_frames

    def _update_angles(self, frame_result: FrameResult) -> Dict[str, HudAngleLine]:
        side = frame_result.shooting_side
        source_keys = {
            "elbow": f"{side}_elbow",
            "knee": f"{side}_knee",
            "trunk": "trunk",
        }
        display: Dict[str, HudAngleLine] = {}

        for short, full_key in source_keys.items():
            raw = frame_result.angles.get(full_key)
            display[short] = self._smooth_angle(short, raw)

        return display

    def _smooth_angle(self, short: str, raw: Optional[AngleResult]) -> HudAngleLine:
        if raw is None or not raw.is_valid or raw.degrees is None:
            invalid_count = self._angle_invalid_counter.get(short, 0) + 1
            self._angle_invalid_counter[short] = invalid_count
            if short in self._angle_last_valid and invalid_count <= self._angle_invalid_hold:
                held_text = str(self._angle_last_valid[short].text)
                return HudAngleLine(
                    text=held_text if held_text.startswith("~") else f"~{held_text}",
                    is_valid=True,
                    is_stable=False,
                )
            return HudAngleLine(text="N/A", is_valid=False, is_stable=False)

        self._angle_invalid_counter[short] = 0
        prev_ema = self._angle_ema.get(short, raw.degrees)
        ema = self._angle_alpha * raw.degrees + (1.0 - self._angle_alpha) * prev_ema
        self._angle_ema[short] = ema
        rounded = int(round(ema))

        shown = self._angle_shown_int.get(short)
        hold = self._angle_hold_counter.get(short, self._angle_min_hold)

        if shown is None:
            self._angle_shown_int[short] = rounded
            self._angle_hold_counter[short] = 0
        elif hold < self._angle_min_hold:
            self._angle_hold_counter[short] = hold + 1
        elif abs(rounded - shown) >= self._angle_step:
            self._angle_shown_int[short] = rounded
            self._angle_hold_counter[short] = 0

        text = str(self._angle_shown_int[short])
        if not raw.is_stable:
            text = f"~{text}"

        line = HudAngleLine(text=text, is_valid=True, is_stable=raw.is_stable)
        self._angle_last_valid[short] = HudAngleLine(
            text=str(self._angle_shown_int[short]),
            is_valid=True,
            is_stable=raw.is_stable,
        )
        return line

    def _update_rules_and_violations(self, frame_result: FrameResult):
        analysis = frame_result.analysis
        if not analysis or analysis.total_count == 0:
            return

        rules_line = f"Rules {analysis.passed_count}/{analysis.total_count}"
        new_violations = [
            HudViolationLine(message=r.message, severity=r.severity)
            for r in analysis.active_rules
            if not r.passed
        ][: self._violation_max]

        if self._violation_hold_remaining > 0:
            self._violation_hold_remaining -= 1
            if rules_line != self._rules_line:
                self._rules_hold = max(self._rules_hold, 8)
        else:
            self._violations = new_violations
            self._violation_hold_remaining = self._violation_hold

        if self._rules_hold > 0:
            self._rules_hold -= 1
        else:
            self._rules_line = rules_line
            self._rules_hold = 12
