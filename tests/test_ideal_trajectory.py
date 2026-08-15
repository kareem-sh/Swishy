"""Focused checks for the 45-degree reference trajectory calculations."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.ideal_trajectory import (
    IdealTrajectoryTracker,
    distance_adaptive_release_angle_deg,
    estimate_release_height_m,
    ideal_physical_trajectory_pixels,
    ideal_trajectory_pixels,
    pose_nose_to_ankle_length_px,
    required_release_speed,
)
from ball.models import BallSnapshot


def _ball(x: float, y: float, frame: int) -> BallSnapshot:
    return BallSnapshot(
        center_xy=(x, y),
        confidence=0.95,
        frame_index=frame,
        timestamp_ms=frame * 100,
    )


def test_ideal_curve_starts_at_release_and_ends_at_rim_center() -> None:
    curve = ideal_trajectory_pixels(
        release_xy=(100.0, 300.0),
        rim_xy=(500.0, 200.0),
        release_angle_deg=45.0,
        samples=81,
    )

    assert curve[0] == (100.0, 300.0)
    assert math.isclose(curve[-1][0], 500.0, abs_tol=1e-9)
    assert math.isclose(curve[-1][1], 200.0, abs_tol=1e-9)
    assert min(y for _, y in curve) < 200.0


def test_required_speed_matches_level_45_degree_solution() -> None:
    speed = required_release_speed(
        horizontal_distance_m=10.0,
        vertical_difference_m=0.0,
        release_angle_deg=45.0,
    )

    assert speed is not None
    assert math.isclose(speed, math.sqrt(9.81 * 10.0), rel_tol=1e-9)


def test_distance_adaptive_angles_match_configured_shot_ranges() -> None:
    assert distance_adaptive_release_angle_deg(1.0) == 65.0
    assert distance_adaptive_release_angle_deg(1.5) == 65.0
    assert distance_adaptive_release_angle_deg(2.25) == 60.0
    assert distance_adaptive_release_angle_deg(3.0) == 55.0
    assert distance_adaptive_release_angle_deg(4.875) == 50.0
    assert distance_adaptive_release_angle_deg(6.75) == 45.0
    assert distance_adaptive_release_angle_deg(8.0) == 45.0


def test_tracker_calculates_angle_speed_and_path_error() -> None:
    tracker = IdealTrajectoryTracker(
        release_angle_deg=45.0,
        samples=21,
        velocity_fit_window_s=0.2,
        velocity_fit_min_points=3,
        comparison_min_points=3,
    )
    rim = (500.0, 200.0)
    radius = 20.0
    curve = ideal_trajectory_pixels((300.0, 260.0), rim, 45.0, 21)

    for frame, (x, y) in enumerate(curve[:8]):
        tracker.update(
            _ball(x, y, frame),
            released_this_frame=frame == 0,
            shot_finished=False,
            rim_center_xy=rim,
            rim_radius=radius,
        )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.observed_release_angle_deg is not None
    assert comparison.observed_release_speed_m_s is not None
    assert comparison.target_release_speed_m_s is not None
    assert comparison.path_rmse_rim_radii is not None
    assert comparison.path_rmse_rim_radii < 1e-9
    assert comparison.observed_point_count == 8
    assert comparison.target_entry_angle_deg is not None
    assert comparison.observed_entry_angle_deg is None
    assert comparison.entry_angle_fit_status == "rim_crossing_unavailable"


def test_reference_curve_follows_a_moving_rim() -> None:
    tracker = IdealTrajectoryTracker()
    tracker.update(
        _ball(100.0, 200.0, 0),
        released_this_frame=True,
        shot_finished=False,
        rim_center_xy=(300.0, 100.0),
        rim_radius=20.0,
    )

    moved_curve = tracker.ideal_screen_curve((330.0, 120.0), 20.0)
    assert moved_curve[0] == (130.0, 220.0)
    assert math.isclose(moved_curve[-1][0], 330.0, abs_tol=1e-9)
    assert math.isclose(moved_curve[-1][1], 120.0, abs_tol=1e-9)


def test_tracker_accepts_release_backtracked_relative_points() -> None:
    tracker = IdealTrajectoryTracker()
    tracker.start_from_relative_points(
        [(-8.0, 5.0, 100), (-7.0, 4.0, 133), (-6.0, 3.2, 166)]
    )

    assert tracker.release_screen_xy((500.0, 300.0), 20.0) == (340.0, 400.0)
    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.observed_point_count == 3


def test_tracker_can_target_top_of_rim_instead_of_geometry_center() -> None:
    tracker = IdealTrajectoryTracker()
    tracker.start_from_relative_points(
        [(-8.0, 5.0, 100), (-7.0, 4.0, 133), (-6.0, 3.2, 166)],
        target_relative_xy=(0.0, -1.5),
    )

    target = tracker.target_screen_xy((500.0, 300.0), 20.0)
    curve = tracker.ideal_screen_curve((500.0, 300.0), 20.0)
    assert target == (500.0, 270.0)
    assert math.isclose(curve[-1][0], 500.0, abs_tol=1e-9)
    assert math.isclose(curve[-1][1], 270.0, abs_tol=1e-9)

    comparison = tracker.comparison()
    assert comparison is not None
    expected_vertical_m = (5.0 - (-1.5)) * (0.457 / 2.0)
    assert math.isclose(
        comparison.vertical_distance_m,
        expected_vertical_m,
        rel_tol=1e-9,
    )


def test_release_height_uses_known_player_scale() -> None:
    height, meters_per_pixel = estimate_release_height_m(
        release_y_px=300.0,
        standing_ankle_y_px=700.0,
        nose_to_ankle_px=644.0,
        player_height_m=1.75,
        nose_to_ankle_height_fraction=0.92,
        ankle_to_floor_height_fraction=0.04,
    )
    assert height is not None
    assert meters_per_pixel is not None
    assert math.isclose(meters_per_pixel, 0.0025, rel_tol=1e-9)
    assert math.isclose(height, 1.07, rel_tol=1e-9)


def test_pose_chain_length_does_not_shrink_when_leg_is_bent() -> None:
    def landmark(x: float, y: float) -> dict:
        return {"x": x, "y": y, "visibility": 1.0}

    straight = {
        "nose": landmark(100, 100),
        "left_shoulder": landmark(90, 150),
        "right_shoulder": landmark(110, 150),
        "left_hip": landmark(90, 250),
        "right_hip": landmark(110, 250),
        "left_knee": landmark(90, 350),
        "right_knee": landmark(110, 350),
        "left_ankle": landmark(90, 450),
        "right_ankle": landmark(110, 450),
    }
    bent = dict(straight)
    bent.update(
        {
            "left_knee": landmark(190, 250),
            "right_knee": landmark(210, 250),
            "left_ankle": landmark(190, 350),
            "right_ankle": landmark(210, 350),
        }
    )
    straight_length = pose_nose_to_ankle_length_px(straight)
    bent_length = pose_nose_to_ankle_length_px(bent)
    assert straight_length is not None
    assert bent_length is not None
    assert math.isclose(straight_length, bent_length, rel_tol=1e-9)


def test_physical_curve_ends_at_the_rim_target() -> None:
    curve = ideal_physical_trajectory_pixels(
        release_xy=(300.0, 400.0),
        target_xy=(500.0, 280.0),
        horizontal_distance_m=2.5,
        vertical_difference_m=0.95,
        release_angle_deg=50.0,
        samples=41,
    )
    assert curve[0] == (300.0, 400.0)
    assert math.isclose(curve[-1][0], 500.0, abs_tol=1e-9)
    assert math.isclose(curve[-1][1], 280.0, abs_tol=1e-9)


def test_small_release_to_rim_height_does_not_explode_metric_arc() -> None:
    curve = ideal_physical_trajectory_pixels(
        release_xy=(300.0, 400.0),
        target_xy=(700.0, 300.0),
        horizontal_distance_m=5.0,
        vertical_difference_m=0.18,
        release_angle_deg=45.0,
        samples=41,
        meters_per_pixel=0.0067,
    )

    assert curve[0] == (300.0, 400.0)
    assert math.isclose(curve[-1][0], 700.0, abs_tol=1e-9)
    assert math.isclose(curve[-1][1], 300.0, abs_tol=1e-9)
    assert min(y for _, y in curve) > 0.0


def test_physical_tracker_uses_release_and_rim_heights_for_velocity() -> None:
    tracker = IdealTrajectoryTracker(rim_height_m=3.05)
    tracker.start_from_relative_points(
        [(-10.0, 5.0, 0), (-9.5, 4.5, 33), (-9.0, 4.0, 66)],
        target_relative_xy=(0.0, -1.0),
        release_height_m=2.10,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
    )
    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.velocity_calibration == "player_height"
    assert math.isclose(comparison.release_height_m, 2.10, rel_tol=1e-9)
    assert math.isclose(comparison.rim_height_m, 3.05, rel_tol=1e-9)
    assert math.isclose(comparison.horizontal_distance_m, 0.5, rel_tol=1e-9)
    assert math.isclose(comparison.vertical_distance_m, 0.95, rel_tol=1e-9)


def test_tracker_can_use_wrist_proxy_as_ideal_release_anchor() -> None:
    tracker = IdealTrajectoryTracker(rim_height_m=3.05)
    tracker.start_from_relative_points(
        [(-5.0, -3.0, 100), (-4.0, -4.0, 133)],
        target_relative_xy=(0.0, -1.0),
        release_height_m=2.10,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
        release_relative_xy=(-10.0, 5.0),
    )
    assert tracker.release_screen_xy((500.0, 300.0), 20.0) == (300.0, 400.0)
    comparison = tracker.comparison()
    assert comparison is not None
    # The velocity fit still counts only the two observed ball points.
    assert comparison.observed_point_count == 2
    assert comparison.release_anchor_source == "wrist_proxy"


def test_release_kinematics_skip_pre_separation_preroll() -> None:
    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.066, velocity_fit_min_points=3
    )
    tracker.start_from_relative_points(
        [
            (-10.0, 5.0, 0),
            (-10.5, 4.8, 33),
            (-9.0, 4.0, 66),
            (-8.0, 3.1, 99),
            (-7.0, 2.3, 132),
        ],
        target_relative_xy=(0.0, 0.0),
        kinematics_start_timestamp_ms=66,
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.observed_release_angle_deg is not None
    assert comparison.observed_release_speed_m_s is not None


def test_release_kinematics_requires_direct_detector_anchors() -> None:
    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.2,
        velocity_fit_min_points=3,
        velocity_fit_min_direct_points=2,
    )
    tracker.start_from_relative_points(
        [
            (-5.0, 3.0, 0, "nanotrack", False),
            (-4.0, 2.0, 100, "nanotrack", False),
            (-3.0, 1.2, 200, "nanotrack", False),
        ],
        target_relative_xy=(0.0, 0.0),
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.observed_release_speed_m_s is None
    assert comparison.early_kinematics_status == "insufficient_direct_detections"
    assert comparison.observed_direct_point_count == 0
    assert comparison.observed_tracked_point_count == 3


def test_release_velocity_window_is_frame_rate_independent() -> None:
    def comparison_for_step(step_ms: int):
        tracker = IdealTrajectoryTracker(
            velocity_fit_window_s=0.2,
            velocity_fit_min_points=3,
            velocity_fit_min_direct_points=2,
        )
        points = []
        for timestamp_ms in range(0, 201, step_ms):
            t = timestamp_ms / 1000.0
            points.append(
                (-5.0 + 10.0 * t, 3.0 - 8.0 * t + 4.0 * t * t, timestamp_ms)
            )
        tracker.start_from_relative_points(
            points,
            target_relative_xy=(0.0, 0.0),
        )
        return tracker.comparison()

    low_fps = comparison_for_step(100)
    high_fps = comparison_for_step(20)

    assert low_fps is not None and high_fps is not None
    assert low_fps.observed_release_angle_deg is not None
    assert high_fps.observed_release_angle_deg is not None
    assert low_fps.observed_release_speed_m_s is not None
    assert high_fps.observed_release_speed_m_s is not None
    assert math.isclose(
        low_fps.observed_release_angle_deg,
        high_fps.observed_release_angle_deg,
        rel_tol=1e-9,
    )
    assert math.isclose(
        low_fps.observed_release_speed_m_s,
        high_fps.observed_release_speed_m_s,
        rel_tol=1e-9,
    )


def _calibrated_launch_comparison(speed_m_s: float, angle_deg: float):
    scale_m_per_radius = 20.0 * 0.0025
    vx_radii_s = (
        speed_m_s * math.cos(math.radians(angle_deg))
        / scale_m_per_radius
    )
    vy_radii_s = (
        speed_m_s * math.sin(math.radians(angle_deg))
        / scale_m_per_radius
    )
    gravity_radii_s2 = 9.81 / scale_m_per_radius
    points = []
    for timestamp_ms in (0, 100, 200):
        t = timestamp_ms / 1000.0
        points.append(
            (
                -84.5 + vx_radii_s * t,
                26.3 - vy_radii_s * t + 0.5 * gravity_radii_s2 * t * t,
                timestamp_ms,
            )
        )

    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.2,
        velocity_fit_min_points=3,
        reachability_vertical_tolerance_m=0.2285,
    )
    tracker.start_from_relative_points(
        points,
        target_relative_xy=(0.0, 0.0),
        release_height_m=1.735,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
        horizontal_distance_m_override=4.225,
    )
    return tracker.comparison()


def test_impossible_early_speed_is_rejected_by_physical_reachability() -> None:
    comparison = _calibrated_launch_comparison(1.689, 79.0)

    assert comparison is not None
    assert comparison.early_kinematics_status == "physically_unreachable_target"
    assert comparison.early_reachability_status == "below_required_height"
    assert comparison.observed_release_angle_deg is None
    assert comparison.observed_release_speed_m_s is None
    assert comparison.predicted_height_at_rim_distance_m is not None
    assert comparison.reachability_height_margin_m is not None
    assert comparison.reachability_height_margin_m < -0.2285
    assert comparison.minimum_reachable_speed_m_s is not None
    assert comparison.minimum_reachable_speed_m_s > 1.689


def test_reachable_early_speed_survives_physical_reachability() -> None:
    comparison = _calibrated_launch_comparison(8.0, 55.0)

    assert comparison is not None
    assert comparison.early_kinematics_status == "accepted"
    assert comparison.early_reachability_status == "accepted"
    assert comparison.observed_release_angle_deg is not None
    assert comparison.observed_release_speed_m_s is not None
    assert math.isclose(
        comparison.observed_release_angle_deg, 55.0, abs_tol=1e-6
    )
    assert math.isclose(
        comparison.observed_release_speed_m_s, 8.0, abs_tol=1e-6
    )


def test_invalid_away_from_rim_window_does_not_report_speed() -> None:
    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.066, velocity_fit_min_points=3
    )
    tracker.start_from_relative_points(
        [
            (-8.0, 5.0, 0),
            (-8.5, 4.8, 33),
            (-9.0, 4.0, 66),
            (-10.0, 3.0, 99),
            (-11.0, 2.0, 132),
        ],
        target_relative_xy=(0.0, 0.0),
        kinematics_start_timestamp_ms=66,
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.observed_release_angle_deg is None
    assert comparison.observed_release_speed_m_s is None


def test_rim_crossing_recovers_launch_when_early_fit_is_invalid() -> None:
    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.066, velocity_fit_min_points=3
    )
    tracker.start_from_relative_points(
        [
            (-8.0, 5.0, 0),
            (-8.5, 4.8, 33),
            (-9.0, 4.0, 66),
            (-10.0, 3.0, 99),
            (-11.0, 2.0, 132),
        ],
        target_relative_xy=(0.0, 0.0),
        release_height_m=2.0,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
        release_relative_xy=(-8.0, 5.0),
        kinematics_start_timestamp_ms=66,
    )
    tracker.update(
        None,
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(500.0, 300.0),
        rim_radius=20.0,
        crossing_xy=(500.0, 300.0),
        crossing_timestamp_ms=1000,
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.early_kinematics_status == "not_moving_toward_rim"
    assert comparison.observed_kinematics_source == "release_to_rim_ballistic"
    assert comparison.observed_release_angle_deg is not None
    assert comparison.observed_release_speed_m_s is not None


def test_release_to_rim_average_speed_uses_crossing_time() -> None:
    tracker = IdealTrajectoryTracker(
        velocity_fit_window_s=0.2, velocity_fit_min_points=3
    )
    tracker.start_from_relative_points(
        [(-10.0, 5.0, 0), (-9.0, 4.0, 100), (-8.0, 3.0, 200)],
        target_relative_xy=(0.0, 0.0),
        kinematics_start_timestamp_ms=0,
    )
    tracker.update(
        None,
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(500.0, 300.0),
        rim_radius=20.0,
        crossing_xy=(500.0, 300.0),
        crossing_timestamp_ms=1000,
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert math.isclose(comparison.release_to_rim_time_s, 1.0, rel_tol=1e-9)
    expected_distance = math.hypot(10.0, 5.0) * (0.457 / 2.0)
    assert math.isclose(
        comparison.release_to_rim_displacement_m,
        expected_distance,
        rel_tol=1e-9,
    )
    expected_path_length = (
        2.0 * math.sqrt(2.0) + math.hypot(8.0, 3.0)
    ) * (0.457 / 2.0)
    assert math.isclose(
        comparison.release_to_rim_path_length_m,
        expected_path_length,
        rel_tol=1e-9,
    )
    assert math.isclose(
        comparison.release_to_rim_average_speed_m_s,
        expected_path_length,
        rel_tol=1e-9,
    )


def test_tracker_rejects_near_rim_release_height_that_explodes_projection() -> None:
    tracker = IdealTrajectoryTracker(
        rim_height_m=3.05,
        minimum_vertical_difference_m=0.25,
    )
    tracker.start_from_relative_points(
        [(-10.0, 1.0, 0), (-9.5, 0.5, 33)],
        target_relative_xy=(0.0, 0.0),
        release_height_m=3.024,
        meters_per_pixel_at_release=0.006234,
        rim_radius_px=35.0,
    )
    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.velocity_calibration == "rim_diameter_fallback"
    assert comparison.release_height_m is None
    assert math.isclose(comparison.rim_height_m, 3.05, rel_tol=1e-9)


def test_trusted_wrist_release_can_be_close_to_rim_height() -> None:
    tracker = IdealTrajectoryTracker(
        rim_height_m=3.05,
        minimum_vertical_difference_m=0.25,
        minimum_wrist_vertical_difference_m=0.05,
    )
    tracker.start_from_relative_points(
        [(-10.0, 1.0, 0), (-9.5, 0.5, 33)],
        target_relative_xy=(0.0, 0.0),
        release_height_m=2.87,
        meters_per_pixel_at_release=0.0065,
        rim_radius_px=35.0,
        release_relative_xy=(-12.0, 2.0),
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert comparison.velocity_calibration == "player_height"
    assert math.isclose(comparison.release_height_m, 2.87, rel_tol=1e-9)
    assert math.isclose(comparison.rim_height_m, 3.05, rel_tol=1e-9)


def test_tracker_freezes_adaptive_angle_from_physical_distance() -> None:
    tracker = IdealTrajectoryTracker(
        release_angle_deg=58.0,
        angle_mode="distance_adaptive",
        rim_height_m=3.05,
    )
    # 45 rim radii * 20 px/radius * 0.0025 m/px = 2.25 m.
    tracker.start_from_relative_points(
        [(-45.0, 5.0, 0), (-44.0, 4.0, 33)],
        target_relative_xy=(0.0, 0.0),
        release_height_m=2.0,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
    )
    comparison = tracker.comparison()
    assert comparison is not None
    assert math.isclose(comparison.horizontal_distance_m, 2.25, rel_tol=1e-9)
    assert math.isclose(comparison.target_release_angle_deg, 60.0, rel_tol=1e-9)

    # Moving/stabilizing the rim does not change the angle during this shot.
    tracker.ideal_screen_curve((700.0, 280.0), 24.0)
    assert tracker.comparison().target_release_angle_deg == 60.0


def test_court_location_overrides_image_distance() -> None:
    tracker = IdealTrajectoryTracker(
        angle_mode="distance_adaptive",
        midrange_angle_deg=60.0,
        three_point_distance_m=5.30,
        three_point_angle_deg=45.0,
    )
    tracker.start_from_relative_points(
        [(-10.0, 5.0, 0), (-9.0, 4.0, 33)],
        target_relative_xy=(0.0, 0.0),
        release_height_m=2.0,
        meters_per_pixel_at_release=0.0025,
        rim_radius_px=20.0,
        horizontal_distance_m_override=4.225,
    )

    comparison = tracker.comparison()
    assert comparison is not None
    assert math.isclose(comparison.horizontal_distance_m, 4.225, rel_tol=1e-9)
    assert comparison.horizontal_distance_source == "fiba_court_location"


def _comparison_for_known_apex(apex_progress: float):
    """Build an exact image-space parabola with a chosen peak position."""
    release_x = -10.0
    release_y = 6.0
    target_x = 0.0
    # For y=A(t-p)^2+y_apex, solve A so y(0)=release_y and
    # y(1)=0. Image y grows downward, so the vertex is the highest point.
    quadratic = release_y / (2.0 * apex_progress - 1.0)
    apex_y = -quadratic * (1.0 - apex_progress) ** 2
    points = []
    for index in range(11):
        progress = index / 10.0
        points.append(
            (
                release_x + (target_x - release_x) * progress,
                quadratic * (progress - apex_progress) ** 2 + apex_y,
                1000 + index * 100,
                "yolo",
                True,
            )
        )

    tracker = IdealTrajectoryTracker(
        release_angle_deg=54.0,
        comparison_min_points=5,
    )
    tracker.start_from_relative_points(
        points,
        target_relative_xy=(target_x, 0.0),
        kinematics_start_timestamp_ms=1000,
        horizontal_distance_m_override=4.225,
    )
    tracker.update(
        None,
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(500.0, 300.0),
        rim_radius=20.0,
        crossing_xy=(500.0, 300.0),
        crossing_timestamp_ms=2000,
    )
    return tracker.comparison()


def test_apex_position_reports_when_and_where_ball_peaked() -> None:
    comparison = _comparison_for_known_apex(0.65)

    assert comparison is not None
    assert comparison.apex_fit_status == "accepted"
    assert comparison.observed_apex_timestamp_ms == 1650
    assert math.isclose(
        comparison.observed_apex_time_from_release_s,
        0.65,
        abs_tol=1e-9,
    )
    assert math.isclose(
        comparison.observed_apex_x_rim_radii,
        -3.5,
        abs_tol=1e-9,
    )
    assert math.isclose(
        comparison.observed_apex_progress,
        0.65,
        abs_tol=1e-9,
    )
    assert math.isclose(
        comparison.observed_apex_distance_m,
        0.65 * 4.225,
        abs_tol=1e-9,
    )
    assert comparison.ideal_apex_timestamp_ms is not None
    assert comparison.ideal_apex_progress is not None
    assert comparison.apex_position_error_progress is not None
    assert comparison.apex_time_error_s is not None
    expected_entry_angle = math.degrees(math.atan2(14.0, 10.0))
    assert comparison.entry_angle_fit_status == "accepted"
    assert math.isclose(
        comparison.observed_entry_angle_deg,
        expected_entry_angle,
        abs_tol=1e-9,
    )
    assert comparison.target_entry_angle_deg is not None
    assert math.isclose(
        comparison.entry_angle_error_deg,
        comparison.observed_entry_angle_deg
        - comparison.target_entry_angle_deg,
        abs_tol=1e-9,
    )


def test_apex_position_error_distinguishes_early_and_late_peaks() -> None:
    early = _comparison_for_known_apex(0.65)
    late = _comparison_for_known_apex(0.95)

    assert early is not None and late is not None
    assert early.apex_position_error_progress is not None
    assert late.apex_position_error_progress is not None
    assert early.apex_position_error_progress < 0.0
    assert late.apex_position_error_progress > 0.0


if __name__ == "__main__":
    test_ideal_curve_starts_at_release_and_ends_at_rim_center()
    test_required_speed_matches_level_45_degree_solution()
    test_distance_adaptive_angles_match_configured_shot_ranges()
    test_tracker_calculates_angle_speed_and_path_error()
    test_reference_curve_follows_a_moving_rim()
    test_tracker_accepts_release_backtracked_relative_points()
    test_tracker_can_target_top_of_rim_instead_of_geometry_center()
    test_release_height_uses_known_player_scale()
    test_pose_chain_length_does_not_shrink_when_leg_is_bent()
    test_physical_curve_ends_at_the_rim_target()
    test_small_release_to_rim_height_does_not_explode_metric_arc()
    test_physical_tracker_uses_release_and_rim_heights_for_velocity()
    test_tracker_can_use_wrist_proxy_as_ideal_release_anchor()
    test_release_kinematics_skip_pre_separation_preroll()
    test_release_kinematics_requires_direct_detector_anchors()
    test_release_velocity_window_is_frame_rate_independent()
    test_impossible_early_speed_is_rejected_by_physical_reachability()
    test_reachable_early_speed_survives_physical_reachability()
    test_invalid_away_from_rim_window_does_not_report_speed()
    test_rim_crossing_recovers_launch_when_early_fit_is_invalid()
    test_release_to_rim_average_speed_uses_crossing_time()
    test_tracker_rejects_near_rim_release_height_that_explodes_projection()
    test_trusted_wrist_release_can_be_close_to_rim_height()
    test_tracker_freezes_adaptive_angle_from_physical_distance()
    test_court_location_overrides_image_distance()
    test_apex_position_reports_when_and_where_ball_peaked()
    test_apex_position_error_distinguishes_early_and_late_peaks()
    print("All ideal trajectory tests passed.")
