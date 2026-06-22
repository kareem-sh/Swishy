"""
Skeleton and angle overlay rendering.

Rendering is decoupled from analysis: this module only displays data
produced by ShotAnalysisPipeline.
"""

import cv2
import numpy as np

from angles.calculator import AngleResult
from pipeline import FrameResult
from phase_detection.phases import PHASE_LABELS

POSE_CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

LANDMARK_INDEX = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _angle_color(result: AngleResult) -> tuple:
    if not result.is_valid:
        return (128, 128, 128)
    if not result.is_stable:
        return (0, 165, 255)
    return (0, 255, 0)


def _format_angle(result: AngleResult) -> str:
    if not result.is_valid or result.degrees is None:
        return "N/A"
    suffix = "~" if not result.is_stable else ""
    return f"{suffix}{int(result.degrees)}"


def _severity_color(severity: str, passed: bool) -> tuple:
    if passed:
        return (0, 255, 0)
    if severity == "error":
        return (0, 0, 255)
    if severity == "warning":
        return (0, 165, 255)
    return (200, 200, 0)


def render_frame(rgb_image: np.ndarray, detection_result, frame_result: FrameResult) -> np.ndarray:
    """Draw skeleton, 3D-computed angles, and pipeline status on the image."""
    annotated = np.copy(rgb_image)

    if not detection_result.pose_landmarks:
        return annotated

    height, width, _ = annotated.shape

    for pose_landmarks in detection_result.pose_landmarks:
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_lm = pose_landmarks[start_idx]
            end_lm = pose_landmarks[end_idx]
            start_pt = (int(start_lm.x * width), int(start_lm.y * height))
            end_pt = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(annotated, start_pt, end_pt, (0, 255, 0), 2)

        for idx, landmark in enumerate(pose_landmarks):
            px, py = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(annotated, (px, py), 4, (255, 0, 0), -1)

        if frame_result.has_pose and frame_result.image_landmarks:
            side = frame_result.shooting_side
            _draw_joint_angle(annotated, frame_result, f"{side}_elbow", frame_result.image_landmarks)
            _draw_joint_angle(annotated, frame_result, f"{side}_knee", frame_result.image_landmarks)
            if "trunk" in frame_result.angles:
                _draw_status_text(annotated, frame_result)

    return annotated


def _draw_joint_angle(image, frame_result: FrameResult, angle_name: str, image_landmarks: dict):
    result = frame_result.angles.get(angle_name)
    if result is None:
        return

    vertex_name = angle_name.split("_", 1)[1] if "_" in angle_name else angle_name
    side = frame_result.shooting_side
    landmark_key = f"{side}_{vertex_name}" if angle_name != "trunk" else None

    if landmark_key and landmark_key in image_landmarks:
        pt = image_landmarks[landmark_key]
        x, y = pt["x"], pt["y"]
    else:
        return

    color = _angle_color(result)
    label = f"{vertex_name.title()}: {_format_angle(result)}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _draw_status_text(image, frame_result: FrameResult):
    phase_label = frame_result.phase_label or PHASE_LABELS.get(frame_result.phase, frame_result.phase)
    lines = [
        "Swichy AI Coach",
        f"Side: {frame_result.shooting_side}",
        f"Phase: {phase_label}",
    ]

    if frame_result.shot_in_progress:
        lines.append("Shot: IN PROGRESS")
    elif frame_result.last_shot_score is not None:
        lines.append(f"Last Shot: {frame_result.last_shot_score}/100")

    elbow = frame_result.angles.get(f"{frame_result.shooting_side}_elbow")
    knee = frame_result.angles.get(f"{frame_result.shooting_side}_knee")
    trunk = frame_result.angles.get("trunk")

    if elbow:
        lines.append(f"Elbow (3D): {_format_angle(elbow)}")
    if knee:
        lines.append(f"Knee (3D): {_format_angle(knee)}")
    if trunk:
        lines.append(f"Trunk: {_format_angle(trunk)}")

    if frame_result.analysis and frame_result.analysis.total_count > 0:
        a = frame_result.analysis
        lines.append(f"Rules: {a.passed_count}/{a.total_count} passed")

    y = 28
    for line in lines:
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26

    if frame_result.display_summary:
        _draw_shot_summary_panel(image, frame_result.display_summary)
    else:
        _draw_rule_feedback(image, frame_result, start_y=y + 10)


def _draw_rule_feedback(image, frame_result: FrameResult, start_y: int):
    if not frame_result.analysis:
        return

    y = start_y
    shown = 0
    max_lines = 4

    for result in frame_result.analysis.active_rules:
        if shown >= max_lines:
            break
        color = _severity_color(result.severity, result.passed)
        prefix = "OK" if result.passed else "!"
        text = f"{prefix} {result.message}"
        if len(text) > 48:
            text = text[:45] + "..."
        cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 22
        shown += 1


def _draw_shot_summary_panel(image, summary):
    h, w, _ = image.shape
    panel_h = min(220, h - 20)
    cv2.rectangle(image, (5, 5), (min(420, w - 5), panel_h), (0, 0, 0), -1)
    cv2.rectangle(image, (5, 5), (min(420, w - 5), panel_h), (255, 255, 255), 1)

    score_color = (0, 255, 0) if summary.score >= 75 else (0, 165, 255) if summary.score >= 60 else (0, 0, 255)
    lines = [
        (f"SHOT #{summary.shot_number}  {summary.grade.upper()}", (255, 255, 255)),
        (f"Score: {summary.score}/100", score_color),
        (f"Rules: {summary.passed_count}/{summary.total_count} passed", (200, 200, 200)),
    ]

    y = 30
    for text, color in lines:
        cv2.putText(image, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y += 28

    cv2.putText(image, "Coach:", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    y += 22
    for tip in summary.coaching_tips[:3]:
        tip_text = tip if len(tip) <= 46 else tip[:43] + "..."
        cv2.putText(image, tip_text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y += 20
