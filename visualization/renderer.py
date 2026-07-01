"""
Skeleton and angle overlay rendering.

Rendering is decoupled from analysis: this module only displays data
produced by ShotAnalysisPipeline.
"""

import cv2
import numpy as np

from angles.calculator import AngleResult
from pipeline import FrameResult
from visualization.hud import draw_hud

POSE_CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def _angle_color(result: AngleResult) -> tuple:
    if not result.is_valid:
        return (128, 128, 128)
    if not result.is_stable:
        return (0, 165, 255)
    return (0, 255, 0)


def render_frame(rgb_image: np.ndarray, detection_result, frame_result: FrameResult) -> np.ndarray:
    """Draw skeleton, compact joint markers, and organized HUD."""
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
            cv2.line(annotated, start_pt, end_pt, (0, 220, 100), 2, cv2.LINE_AA)

        for landmark in pose_landmarks:
            px, py = int(landmark.x * width), int(landmark.y * height)
            cv2.circle(annotated, (px, py), 4, (255, 120, 60), -1, cv2.LINE_AA)

        if frame_result.has_pose and frame_result.image_landmarks:
            side = frame_result.shooting_side
            _draw_shooting_chain_highlight(annotated, frame_result, side, frame_result.image_landmarks)

    if frame_result.has_pose:
        draw_hud(annotated, frame_result, frame_result.hud_display)

    return annotated


def _draw_shooting_chain_highlight(image, frame_result: FrameResult, side: str, image_landmarks: dict):
    """Highlight shooting-side joints without duplicating HUD text."""
    chain = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
    points = []
    for name in chain:
        if name in image_landmarks:
            pt = image_landmarks[name]
            points.append((pt["x"], pt["y"]))

    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (255, 200, 0), 3, cv2.LINE_AA)

    for name in (f"{side}_elbow", f"{side}_knee"):
        result = frame_result.angles.get(name)
        key = name
        if result is None or key not in image_landmarks:
            continue
        pt = image_landmarks[key]
        x, y = pt["x"], pt["y"]
        color = _angle_color(result)
        cv2.circle(image, (x, y), 8, color, 2, cv2.LINE_AA)
