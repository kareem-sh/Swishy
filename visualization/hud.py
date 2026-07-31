"""Organized on-screen HUD panels for live/video overlay."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from angles.calculator import AngleResult
from feedback.models import ShotSummary
from visualization.hud_display import HudDisplay, HudViolationLine
from pipeline import FrameResult


def _blend_panel(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, alpha: float = 0.82):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 22), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (70, 70, 80), 1)


def _put_line(image, text: str, x: int, y: int, scale: float, color: Tuple[int, int, int], thickness: int = 1):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _format_angle(result: AngleResult) -> str:
    if not result.is_valid or result.degrees is None:
        return "N/A"
    prefix = "~" if not result.is_stable else ""
    return f"{prefix}{int(result.degrees)}"


def _angle_color(result: AngleResult) -> Tuple[int, int, int]:
    if not result.is_valid:
        return (140, 140, 140)
    if not result.is_stable:
        return (0, 165, 255)
    return (80, 220, 120)


def draw_hud(image: np.ndarray, frame_result: FrameResult, hud_display: Optional[HudDisplay] = None):
    """Draw structured coaching HUD on top of the skeleton overlay."""
    h, w, _ = image.shape

    if frame_result.display_summary:
        _draw_shot_summary_card(image, frame_result.display_summary, w, h)
        _draw_top_bar(image, frame_result, w, compact=True, hud_display=hud_display)
        return

    _draw_top_bar(image, frame_result, w, compact=False, hud_display=hud_display)
    _draw_metrics_panel(image, frame_result, w, h, hud_display=hud_display)
    _draw_coaching_strip(image, frame_result, w, h, hud_display=hud_display)


def _draw_top_bar(image, frame_result: FrameResult, w: int, compact: bool, hud_display: Optional[HudDisplay] = None):
    bar_h = 44
    _blend_panel(image, 0, 0, w, bar_h, alpha=0.88)

    phase = (hud_display.phase_label if hud_display else None) or frame_result.phase_label or frame_result.phase
    _put_line(image, "SWICHY", 14, 28, 0.62, (255, 200, 60), 2)
    _put_line(image, f"|  {phase}", 108, 28, 0.58, (230, 230, 230), 1)

    if frame_result.shot_in_progress:
        status = "SHOT IN PROGRESS"
        color = (80, 200, 255)
    elif frame_result.last_shot_score is not None:
        status = f"LAST SHOT  {frame_result.last_shot_score}/100"
        color = (120, 255, 120) if frame_result.last_shot_score >= 75 else (80, 180, 255)
    else:
        status = "READY"
        color = (180, 180, 180)

    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    _put_line(image, status, w - status_size[0] - 14, 28, 0.55, color, 1)

    if frame_result.capture_warning:
        warn = frame_result.capture_warning
        if len(warn) > 52:
            warn = warn[:49] + "..."
        _put_line(image, warn, 14, bar_h - 6, 0.38, (0, 180, 255), 1)

    if not compact:
        rules = (hud_display.rules_line if hud_display and hud_display.rules_line else None)
        if not rules and frame_result.analysis and frame_result.analysis.total_count > 0:
            a = frame_result.analysis
            rules = f"Rules {a.passed_count}/{a.total_count}"
        if rules:
            rules_size = cv2.getTextSize(rules, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            mid_x = (w - rules_size[0]) // 2
            _put_line(image, rules, mid_x, 28, 0.48, (200, 200, 200), 1)


def _draw_metrics_panel(image, frame_result: FrameResult, w: int, h: int, hud_display: Optional[HudDisplay] = None):
    panel_w = 188
    panel_h = 118
    x1 = w - panel_w - 10
    y1 = 54
    _blend_panel(image, x1, y1, w - 10, y1 + panel_h)

    _put_line(image, "ANGLES (3D)", x1 + 12, y1 + 22, 0.48, (255, 200, 60), 1)

    side = frame_result.shooting_side
    if hud_display and hud_display.angles:
        rows = [
            ("Elbow", hud_display.angles.get("elbow")),
            ("Knee", hud_display.angles.get("knee")),
            ("Trunk", hud_display.angles.get("trunk")),
        ]
        y = y1 + 46
        for label, line in rows:
            if line is None:
                continue
            color = (140, 140, 140) if not line.is_valid else (0, 165, 255) if not line.is_stable else (80, 220, 120)
            _put_line(image, f"{label:<6}", x1 + 12, y, 0.46, (170, 170, 170), 1)
            _put_line(image, line.text, x1 + 90, y, 0.52, color, 1)
            y += 24
    else:
        rows = [
            ("Elbow", frame_result.angles.get(f"{side}_elbow")),
            ("Knee", frame_result.angles.get(f"{side}_knee")),
            ("Trunk", frame_result.angles.get("trunk")),
        ]
        y = y1 + 46
        for label, result in rows:
            if result is None:
                continue
            _put_line(image, f"{label:<6}", x1 + 12, y, 0.46, (170, 170, 170), 1)
            _put_line(image, _format_angle(result), x1 + 90, y, 0.52, _angle_color(result), 1)
            y += 24

    _put_line(image, f"Side: {side}", x1 + 12, y1 + panel_h - 10, 0.42, (140, 140, 150), 1)


def _draw_coaching_strip(image, frame_result: FrameResult, w: int, h: int, hud_display: Optional[HudDisplay] = None):
    violations = []
    if hud_display and hud_display.violations:
        violations = hud_display.violations
    elif frame_result.analysis:
        violations = [r for r in frame_result.analysis.active_rules if not r.passed]

    if not violations:
        return

    max_items = 3
    strip_h = 28 + max_items * 24 + 16
    y1 = h - strip_h - 10
    _blend_panel(image, 10, y1, min(w - 10, 520), h - 10)

    _put_line(image, "FIX NOW", 22, y1 + 24, 0.5, (0, 140, 255), 1)

    y = y1 + 48
    for rule in violations[:max_items]:
        if isinstance(rule, HudViolationLine):
            severity = rule.severity
            message = rule.message
        else:
            severity = rule.severity
            message = rule.message
        icon = "!" if severity == "error" else ">"
        color = (80, 80, 255) if severity == "error" else (0, 165, 255)
        text = message if len(message) <= 52 else message[:49] + "..."
        _put_line(image, f"{icon}  {text}", 22, y, 0.46, color, 1)
        y += 24


def _draw_shot_summary_card(image, summary: ShotSummary, w: int, h: int):
    card_w = min(460, w - 40)
    card_h = min(230, 102 + len(summary.coaching_tips[:3]) * 22)
    x1 = (w - card_w) // 2
    y1 = h - card_h - 24
    _blend_panel(image, x1, y1, x1 + card_w, y1 + card_h, alpha=0.9)

    score_color = (80, 220, 120) if summary.score >= 75 else (0, 165, 255) if summary.score >= 60 else (80, 80, 255)
    _put_line(image, f"SHOT #{summary.shot_number}  |  {summary.grade.upper()}", x1 + 16, y1 + 28, 0.58, (255, 255, 255), 1)
    _put_line(image, f"{summary.score}/100", x1 + card_w - 72, y1 + 28, 0.62, score_color, 2)
    _put_line(
        image,
        f"Rules passed: {summary.passed_count}/{summary.total_count}",
        x1 + 16,
        y1 + 52,
        0.46,
        (180, 180, 180),
        1,
    )

    outcome_text = "UNKNOWN"
    if summary.outcome is not None:
        outcome_text = summary.outcome.result.upper()
    outcome_color = (
        (80, 220, 120)
        if outcome_text == "MADE"
        else (80, 80, 255) if outcome_text == "MISSED" else (0, 180, 255)
    )
    _put_line(
        image,
        f"Basket: {outcome_text}",
        x1 + 16,
        y1 + 76,
        0.48,
        outcome_color,
        1,
    )

    _put_line(image, "Coach says:", x1 + 16, y1 + 100, 0.46, (255, 200, 60), 1)
    y = y1 + 122
    for tip in summary.coaching_tips[:3]:
        tip_text = tip if len(tip) <= 58 else tip[:55] + "..."
        _put_line(image, f"- {tip_text}", x1 + 16, y, 0.44, (230, 230, 230), 1)
        y += 22
