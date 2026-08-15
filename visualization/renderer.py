"""
Skeleton and angle overlay rendering.

Rendering is decoupled from analysis: this module only displays data
produced by ShotAnalysisPipeline.
"""

import cv2
import numpy as np

from angles.calculator import AngleResult
from pipeline import FrameResult
from utils.config_loader import load_yaml
from visualization.hud import draw_hud

POSE_CONNECTIONS = [
    (0, 11), (0, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 19), (16, 20),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
]

# BGR colors on an RGB canvas (OpenCV draw calls before RGB→BGR convert in modes)
_BALL_COLOR = (255, 140, 0)   # orange-ish in RGB
_PREDICTED_BALL_COLOR = (255, 220, 0)
_RIM_COLOR = (0, 220, 80)     # green in RGB
_RIM_GEOMETRY_COLOR = (70, 150, 255)
_FITTED_TRAJECTORY_COLOR = (40, 255, 170)
_IDEAL_TRAJECTORY_COLOR = (80, 190, 255)
_DISPLAY_CONFIG = load_yaml("display.yaml")
_SHOW_BALL_OVERLAY = bool(_DISPLAY_CONFIG.get("show_ball_overlay", True))
_TRAJECTORY_CONFIG = _DISPLAY_CONFIG.get("trajectory_overlay", {})
_SHOW_OBSERVED_TRAJECTORY = bool(
    _TRAJECTORY_CONFIG.get("show_observed", True)
)
_SHOW_OBSERVED_POLYLINE = bool(
    _TRAJECTORY_CONFIG.get("show_observed_polyline", False)
)
_SHOW_FITTED_TRAJECTORY = bool(
    _TRAJECTORY_CONFIG.get("show_fitted", True)
)
_SHOW_IDEAL_TRAJECTORY = bool(
    _TRAJECTORY_CONFIG.get("show_ideal", True)
)
_OBSERVED_TRAJECTORY_THICKNESS = max(
    1, int(_TRAJECTORY_CONFIG.get("observed_thickness", 3))
)
_OBSERVED_POINT_RADIUS = max(
    1, int(_TRAJECTORY_CONFIG.get("observed_point_radius", 2))
)
_FITTED_TRAJECTORY_THICKNESS = max(
    1, int(_TRAJECTORY_CONFIG.get("fitted_thickness", 3))
)
_IDEAL_TRAJECTORY_THICKNESS = max(
    1, int(_TRAJECTORY_CONFIG.get("ideal_thickness", 2))
)


def _draw_observed_ball_trajectory(
    annotated: np.ndarray,
    frame_result: FrameResult,
) -> None:
    """Draw measured post-release points without bridging tracking gaps."""
    if not _SHOW_OBSERVED_TRAJECTORY:
        return

    ideal = frame_result.ideal_ball_path
    if _SHOW_IDEAL_TRAJECTORY and len(ideal) >= 2:
        ideal_points = np.asarray(ideal, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated,
            [ideal_points],
            False,
            _IDEAL_TRAJECTORY_COLOR,
            _IDEAL_TRAJECTORY_THICKNESS,
            cv2.LINE_AA,
        )
        target = frame_result.ideal_rim_target_xy
        if target is not None:
            cv2.circle(
                annotated,
                (int(round(target[0])), int(round(target[1]))),
                5,
                _IDEAL_TRAJECTORY_COLOR,
                2,
                cv2.LINE_AA,
            )

    fitted = frame_result.fitted_observed_ball_path
    if _SHOW_FITTED_TRAJECTORY and len(fitted) >= 2:
        fitted_points = np.asarray(fitted, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            annotated,
            [fitted_points],
            False,
            _FITTED_TRAJECTORY_COLOR,
            _FITTED_TRAJECTORY_THICKNESS,
            cv2.LINE_AA,
        )

    for segment in frame_result.observed_ball_path_segments:
        if not segment:
            continue
        points = np.asarray(segment, dtype=np.int32).reshape((-1, 1, 2))
        if _SHOW_OBSERVED_POLYLINE and len(segment) >= 2:
            cv2.polylines(
                annotated,
                [points],
                False,
                _BALL_COLOR,
                _OBSERVED_TRAJECTORY_THICKNESS,
                cv2.LINE_AA,
            )
        for x, y in points[:, 0, :]:
            cv2.circle(
                annotated,
                (int(x), int(y)),
                _OBSERVED_POINT_RADIUS,
                _BALL_COLOR,
                -1,
                cv2.LINE_AA,
            )


def _angle_color(result: AngleResult) -> tuple:
    if not result.is_valid:
        return (128, 128, 128)
    if not result.is_stable:
        return (0, 165, 255)
    return (0, 255, 0)


def _draw_ball_rim(annotated: np.ndarray, frame_result: FrameResult) -> None:
    """Overlay ball + rim boxes from the custom basketball YOLO."""
    if not _SHOW_BALL_OVERLAY:
        return
    if frame_result.rim is not None:
        x1, y1, x2, y2 = map(int, frame_result.rim.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _RIM_COLOR, 2, cv2.LINE_AA)
        cv2.circle(
            annotated,
            (int(frame_result.rim.x), int(frame_result.rim.y)),
            4,
            _RIM_COLOR,
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"rim {frame_result.rim.confidence:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _RIM_COLOR,
            2,
            cv2.LINE_AA,
        )

    rim_center = frame_result.stabilized_rim_center_xy
    rim_radius = frame_result.stabilized_rim_inner_radius
    if rim_center is not None and rim_radius is not None:
        rim_x, rim_y = map(int, rim_center)
        radius = max(1, int(rim_radius))
        cv2.line(
            annotated,
            (rim_x - radius, rim_y),
            (rim_x + radius, rim_y),
            _RIM_GEOMETRY_COLOR,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            annotated,
            (rim_x, rim_y),
            3,
            _RIM_GEOMETRY_COLOR,
            -1,
            cv2.LINE_AA,
        )

    if frame_result.rim_crossing_xy is not None:
        crossing = tuple(map(int, frame_result.rim_crossing_xy))
        if frame_result.ball_state in ("crossed_inside", "made"):
            crossing_color = (0, 230, 80)
        elif frame_result.ball_state == "rim_contact":
            crossing_color = (255, 210, 0)
        else:
            crossing_color = (255, 70, 70)
        cv2.circle(annotated, crossing, 7, crossing_color, 2, cv2.LINE_AA)

    state_text = (
        f"Ball: {frame_result.ball_state.upper()} | "
        f"{frame_result.ball_tracking_status.upper()} | "
        f"{frame_result.ball_measurement_source.upper()}"
    )
    if frame_result.shot_outcome is not None:
        state_text += f" | {frame_result.shot_outcome.result.upper()}"
    cv2.putText(
        annotated,
        state_text,
        (10, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Prefer smoothed track center when available
    ball = frame_result.ball
    snap = frame_result.ball_snapshot
    if ball is None and snap is None:
        return

    if ball is not None:
        x1, y1, x2, y2 = map(int, ball.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _BALL_COLOR, 2, cv2.LINE_AA)
        conf = ball.confidence
        label_xy = (x1, max(18, y1 - 8))
    else:
        conf = snap.confidence
        label_xy = (int(snap.x) + 8, int(snap.y) - 8)

    cx = int(snap.x) if snap is not None else int(ball.x)
    cy = int(snap.y) if snap is not None else int(ball.y)
    center_color = (
        _PREDICTED_BALL_COLOR
        if frame_result.ball_tracking_status == "predicted"
        else _BALL_COLOR
    )
    cv2.circle(annotated, (cx, cy), 5, center_color, -1, cv2.LINE_AA)
    cv2.putText(
        annotated,
        f"ball {conf:.2f}",
        label_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        _BALL_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_ball_overlays(
    rgb_image: np.ndarray,
    frame_result: FrameResult,
) -> np.ndarray:
    """Draw every ball/rim/trajectory layer used by video mode.

    Keeping this separate from pose rendering lets offline replay reuse the
    exact same basketball visualization without running YOLO or NanoTrack a
    second time.
    """
    _draw_observed_ball_trajectory(rgb_image, frame_result)
    _draw_ball_rim(rgb_image, frame_result)
    return rgb_image


def render_frame(rgb_image: np.ndarray, detection_result, frame_result: FrameResult) -> np.ndarray:
    """Draw ball/rim, skeleton, compact joint markers, and organized HUD."""
    annotated = np.copy(rgb_image)
    draw_ball_overlays(annotated, frame_result)

    if not detection_result or not getattr(detection_result, "pose_landmarks", None):
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
    chain = [
        f"{side}_shoulder",
        f"{side}_elbow",
        f"{side}_wrist",
        f"{side}_index",
    ]
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
