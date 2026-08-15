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
_DISPLAY_CONFIG = load_yaml("display.yaml")
_SHOW_BALL_OVERLAY = bool(_DISPLAY_CONFIG.get("show_ball_overlay", True))
_SHOW_SHOOTER_OVERLAY = bool(_DISPLAY_CONFIG.get("show_shooter_overlay", True))
_SHOW_COURT_OVERLAY = bool(_DISPLAY_CONFIG.get("show_court_overlay", True))
_SHOOTER_COLOR = (255, 80, 220)
_SHOOTER_FEET_COLOR = (255, 255, 80)
_COURT_OUTLINE_COLOR = (80, 200, 255)
_COURT_AXIS_COLOR = (120, 220, 255)
_OTHER_POSE_COLOR = (90, 90, 90)
_OTHER_JOINT_COLOR = (120, 120, 120)
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
_OBSERVED_TRAJECTORY_THICKNESS = max(
    1, int(_TRAJECTORY_CONFIG.get("observed_thickness", 3))
)
_OBSERVED_POINT_RADIUS = max(
    1, int(_TRAJECTORY_CONFIG.get("observed_point_radius", 2))
)
_FITTED_TRAJECTORY_THICKNESS = max(
    1, int(_TRAJECTORY_CONFIG.get("fitted_thickness", 3))
)


def _draw_observed_ball_trajectory(
    annotated: np.ndarray,
    frame_result: FrameResult,
) -> None:
    """Draw measured post-release points without bridging tracking gaps."""
    if not _SHOW_OBSERVED_TRAJECTORY:
        return

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
        f"{frame_result.ball_tracking_status.upper()}"
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


def _draw_court_overlay(annotated: np.ndarray, frame_result: FrameResult) -> None:
    if not _SHOW_COURT_OVERLAY:
        return

    h, w, _ = annotated.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        f"COURT: {frame_result.court_calibration_state}",
    ]
    if frame_result.court_inlier_count is not None:
        lines.append(f"INLIERS: {frame_result.court_inlier_count}")
    if frame_result.court_reprojection_error_px is not None:
        lines.append(f"REPROJ: {frame_result.court_reprojection_error_px:.1f} px")

    x = w - 220
    y = 24
    for line in lines:
        cv2.putText(annotated, line, (x, y), font, 0.48, (220, 220, 230), 1, cv2.LINE_AA)
        y += 18

    if not frame_result.court_outline_px:
        return

    outline = np.asarray(frame_result.court_outline_px, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(annotated, [outline], True, _COURT_OUTLINE_COLOR, 2, cv2.LINE_AA)

    if frame_result.court_origin_px is not None:
        origin = tuple(map(int, frame_result.court_origin_px))
        cv2.circle(annotated, origin, 5, _COURT_AXIS_COLOR, -1, cv2.LINE_AA)
    if frame_result.court_x_axis_px is not None and frame_result.court_origin_px is not None:
        cv2.arrowedLine(
            annotated,
            tuple(map(int, frame_result.court_origin_px)),
            tuple(map(int, frame_result.court_x_axis_px)),
            _COURT_AXIS_COLOR,
            2,
            tipLength=0.2,
        )
    if frame_result.court_y_axis_px is not None and frame_result.court_origin_px is not None:
        cv2.arrowedLine(
            annotated,
            tuple(map(int, frame_result.court_origin_px)),
            tuple(map(int, frame_result.court_y_axis_px)),
            (180, 255, 180),
            2,
            tipLength=0.2,
        )


def _draw_shooter_overlay(annotated: np.ndarray, frame_result: FrameResult) -> None:
    if not _SHOW_SHOOTER_OVERLAY:
        return

    holder = frame_result.shooter or frame_result.ball_holder
    if holder is None:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    if holder.feet_midpoint_xy is not None:
        feet = tuple(map(int, holder.feet_midpoint_xy))
        cv2.circle(annotated, feet, 8, _SHOOTER_FEET_COLOR, 2, cv2.LINE_AA)
        cv2.circle(annotated, feet, 3, _SHOOTER_FEET_COLOR, -1, cv2.LINE_AA)

    if holder.bbox_xyxy is not None:
        x1, y1, x2, y2 = map(int, holder.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), _SHOOTER_COLOR, 2, cv2.LINE_AA)

    lines = [
        f"SHOOTER ID: {holder.player_id}",
        f"CONF: {holder.confidence:.2f}",
        f"STATE: {holder.shooter_state}",
    ]
    if holder.court_position is not None:
        x_m, y_m, z_m = holder.court_position
        lines.extend(
            [
                "COURT:",
                f"X = {x_m:+.2f} m",
                f"Y = {y_m:.2f} m",
                f"Z = {z_m:.2f} m",
            ]
        )
        if holder.distance_to_hoop_m is not None:
            lines.append(f"DIST = {holder.distance_to_hoop_m:.2f} m")

    x, y = 10, 120
    for line in lines:
        cv2.putText(annotated, line, (x, y), font, 0.50, (245, 245, 250), 1, cv2.LINE_AA)
        y += 18

    if frame_result.shooter_release is not None:
        release = frame_result.shooter_release
        cv2.putText(
            annotated,
            "SHOT RELEASED",
            (x, y + 4),
            font,
            0.58,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )


def render_frame(rgb_image: np.ndarray, detection_result, frame_result: FrameResult) -> np.ndarray:
    """Draw ball/rim, skeleton, compact joint markers, and organized HUD."""
    annotated = np.copy(rgb_image)
    _draw_observed_ball_trajectory(annotated, frame_result)
    _draw_ball_rim(annotated, frame_result)
    _draw_court_overlay(annotated, frame_result)

    if not detection_result or not getattr(detection_result, "pose_landmarks", None):
        _draw_shooter_overlay(annotated, frame_result)
        return annotated

    height, width, _ = annotated.shape
    selected_index = frame_result.selected_pose_index

    for pose_index, pose_landmarks in enumerate(detection_result.pose_landmarks):
        is_selected = pose_index == selected_index
        line_color = _SHOOTER_COLOR if is_selected else _OTHER_POSE_COLOR
        joint_color = (255, 120, 60) if is_selected else _OTHER_JOINT_COLOR
        line_thickness = 3 if is_selected else 1
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_lm = pose_landmarks[start_idx]
            end_lm = pose_landmarks[end_idx]
            start_pt = (int(start_lm.x * width), int(start_lm.y * height))
            end_pt = (int(end_lm.x * width), int(end_lm.y * height))
            cv2.line(annotated, start_pt, end_pt, line_color, line_thickness, cv2.LINE_AA)

        for landmark in pose_landmarks:
            px, py = int(landmark.x * width), int(landmark.y * height)
            radius = 5 if is_selected else 3
            cv2.circle(annotated, (px, py), radius, joint_color, -1, cv2.LINE_AA)

        if frame_result.has_pose and is_selected and frame_result.image_landmarks:
            side = frame_result.shooting_side
            _draw_shooting_chain_highlight(annotated, frame_result, side, frame_result.image_landmarks)

    _draw_shooter_overlay(annotated, frame_result)

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
