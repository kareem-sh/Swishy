"""Capture key frames during a shot for detailed reporting."""

from typing import Dict, List, Tuple

import numpy as np

from analysis.models import RuleResult
from feedback.report_models import KeyFrame
from phase_detection.phases import PHASE_LABELS
from pipeline import FrameResult


def _format_timestamp(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _violation_distance(rule: RuleResult) -> float:
    if rule.measured_value is None:
        return 0.0
    val = rule.measured_value
    if rule.min_value is not None and val < rule.min_value:
        return rule.min_value - val
    if rule.max_value is not None and val > rule.max_value:
        return val - rule.max_value
    return 0.0


def _angles_summary(frame_result: FrameResult) -> str:
    side = frame_result.shooting_side
    parts = []
    for key in (f"{side}_knee", f"{side}_elbow", f"{side}_shoulder", f"{side}_hip", "trunk"):
        angle = frame_result.angles.get(key)
        if angle and angle.is_valid and angle.degrees is not None:
            short = key.replace(f"{side}_", "")
            parts.append(f"{short}={angle.degrees:.0f}°")
    return ", ".join(parts)


class KeyFrameCapture:
    """Select and store the most informative frames from one shot."""

    PRIORITY_PHASES = ("loading", "ball_lift", "release", "follow_through", "landing")

    def __init__(self, max_frames: int = 12):
        self.max_frames = max_frames
        self._phase_captured: set = set()
        self._violation_best: Dict[str, Tuple] = {}
        self._captured: List[Tuple[KeyFrame, np.ndarray]] = []

    def consider(
        self,
        frame_index: int,
        annotated_bgr: np.ndarray,
        frame_result: FrameResult,
    ):
        if not frame_result.has_pose or not frame_result.shot_in_progress:
            return

        phase = frame_result.phase
        violations = frame_result.analysis.violations if frame_result.analysis else []

        for rule in violations:
            dist = _violation_distance(rule)
            prev = self._violation_best.get(rule.rule_id)
            if prev is None or dist > prev[0]:
                self._violation_best[rule.rule_id] = (
                    dist,
                    frame_index,
                    frame_result,
                    annotated_bgr.copy(),
                    rule,
                )

        if phase not in self._phase_captured and phase in self.PRIORITY_PHASES:
            self._phase_captured.add(phase)
            kf = self._make_key_frame(
                frame_index,
                frame_result,
                capture_reason=f"Key phase: {PHASE_LABELS.get(phase, phase)}",
                issues=[v.message for v in violations],
                rule_ids=[v.rule_id for v in violations],
            )
            self._captured.append((kf, annotated_bgr.copy()))

    def finalize(self) -> List[Tuple[KeyFrame, np.ndarray]]:
        frames: List[Tuple[KeyFrame, np.ndarray]] = list(self._captured)
        existing_rule_ids = {rid for kf, _ in frames for rid in kf.rule_ids}

        for rule_id, (_, frame_index, fr, img, rule) in self._violation_best.items():
            if rule_id in existing_rule_ids:
                continue
            kf = self._make_key_frame(
                frame_index,
                fr,
                capture_reason=f"Issue: {rule.name}",
                issues=[rule.message],
                rule_ids=[rule_id],
            )
            frames.append((kf, img))

        frames.sort(key=lambda pair: (pair[0].timestamp_ms, pair[0].frame_index))
        return frames[: self.max_frames]

    def _make_key_frame(
        self,
        frame_index: int,
        frame_result: FrameResult,
        capture_reason: str,
        issues: List[str],
        rule_ids: List[str],
    ) -> KeyFrame:
        phase = frame_result.phase
        return KeyFrame(
            frame_index=frame_index,
            timestamp_ms=frame_result.timestamp_ms,
            timestamp_label=_format_timestamp(frame_result.timestamp_ms),
            phase=phase,
            phase_label=frame_result.phase_label or PHASE_LABELS.get(phase, phase),
            image_filename="",
            capture_reason=capture_reason,
            issues=issues,
            rule_ids=rule_ids,
            angles_summary=_angles_summary(frame_result),
        )
