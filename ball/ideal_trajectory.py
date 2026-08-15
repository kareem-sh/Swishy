"""45-degree reference trajectory and measured-flight comparisons.

The calculations use rim-radius units internally. That makes the geometry
independent of video resolution and keeps it aligned when a handheld camera
moves, provided the stabilized rim position and size remain available.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ball.models import BallSnapshot

Point = Tuple[float, float]


@dataclass(frozen=True)
class TrajectoryComparison:
    """Measured values only; no coaching or score is produced here."""

    target_release_angle_deg: float
    observed_release_angle_deg: Optional[float]
    release_angle_error_deg: Optional[float]
    target_release_speed_m_s: Optional[float]
    observed_release_speed_m_s: Optional[float]
    release_speed_error_m_s: Optional[float]
    target_entry_angle_deg: Optional[float]
    observed_entry_angle_deg: Optional[float]
    entry_angle_error_deg: Optional[float]
    entry_angle_fit_status: str
    observed_kinematics_source: str
    early_kinematics_status: str
    early_reachability_status: str
    predicted_height_at_rim_distance_m: Optional[float]
    reachability_height_margin_m: Optional[float]
    minimum_reachable_speed_m_s: Optional[float]
    reachability_vertical_tolerance_m: float
    release_height_m: Optional[float]
    rim_height_m: Optional[float]
    horizontal_distance_m: Optional[float]
    horizontal_distance_source: str
    vertical_distance_m: Optional[float]
    straight_line_distance_m: Optional[float]
    release_to_rim_time_s: Optional[float]
    release_to_rim_displacement_m: Optional[float]
    release_to_rim_path_length_m: Optional[float]
    release_to_rim_average_speed_m_s: Optional[float]
    meters_per_pixel_at_release: Optional[float]
    velocity_calibration: str
    release_anchor_source: str
    path_rmse_rim_radii: Optional[float]
    observed_apex_height_rim_radii: Optional[float]
    ideal_apex_height_rim_radii: Optional[float]
    apex_height_error_rim_radii: Optional[float]
    apex_fit_status: str
    observed_apex_timestamp_ms: Optional[int]
    ideal_apex_timestamp_ms: Optional[int]
    observed_apex_time_from_release_s: Optional[float]
    ideal_apex_time_from_release_s: Optional[float]
    apex_time_error_s: Optional[float]
    observed_apex_x_rim_radii: Optional[float]
    ideal_apex_x_rim_radii: Optional[float]
    observed_apex_progress: Optional[float]
    ideal_apex_progress: Optional[float]
    apex_position_error_progress: Optional[float]
    observed_apex_distance_m: Optional[float]
    ideal_apex_distance_m: Optional[float]
    apex_position_error_m: Optional[float]
    rim_crossing_error_rim_radii: Optional[float]
    observed_point_count: int
    observed_direct_point_count: int
    observed_tracked_point_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _RelativePoint:
    x: float
    y: float
    timestamp_ms: int
    measurement_source: str = "unknown"
    is_direct_observation: bool = True


@dataclass(frozen=True)
class _ApexComparison:
    """Fitted observed peak and the corresponding ideal peak."""

    fit_status: str
    observed_height_rim_radii: Optional[float]
    ideal_height_rim_radii: Optional[float]
    observed_timestamp_ms: Optional[int]
    ideal_timestamp_ms: Optional[int]
    observed_time_from_release_s: Optional[float]
    ideal_time_from_release_s: Optional[float]
    observed_x_rim_radii: Optional[float]
    ideal_x_rim_radii: Optional[float]
    observed_progress: Optional[float]
    ideal_progress: Optional[float]
    observed_distance_m: Optional[float]
    ideal_distance_m: Optional[float]


@dataclass(frozen=True)
class _ObservedFlightFit:
    """Robust rim-relative x(t), y(t) fit for one release-to-rim flight."""

    status: str
    release_timestamp_ms: int
    first_time_s: Optional[float] = None
    last_time_s: Optional[float] = None
    x_coefficients: Optional[Tuple[float, float]] = None
    y_coefficients: Optional[Tuple[float, float, float]] = None


def ideal_trajectory_pixels(
    release_xy: Point,
    rim_xy: Point,
    release_angle_deg: float = 45.0,
    samples: int = 80,
) -> List[Point]:
    """Return a parabola that starts at release and ends at a rim target."""
    release_x, release_y = map(float, release_xy)
    rim_x, rim_y = map(float, rim_xy)
    horizontal_distance = abs(rim_x - release_x)
    if horizontal_distance <= 1e-6:
        return []

    direction = 1.0 if rim_x >= release_x else -1.0
    height_to_rim = release_y - rim_y
    theta = math.radians(float(release_angle_deg))
    curvature = (
        horizontal_distance * math.tan(theta) - height_to_rim
    ) / (horizontal_distance * horizontal_distance)

    return [
        (
            release_x + direction * distance,
            release_y
            - (
                distance * math.tan(theta)
                - curvature * distance * distance
            ),
        )
        for distance in np.linspace(
            0.0, horizontal_distance, max(2, int(samples))
        )
    ]


def required_release_speed(
    horizontal_distance_m: float,
    vertical_difference_m: float,
    release_angle_deg: float = 45.0,
    gravity_m_s2: float = 9.81,
) -> Optional[float]:
    """Solve projectile motion for speed at a chosen release angle."""
    distance = float(horizontal_distance_m)
    if distance <= 0.0:
        return None
    theta = math.radians(float(release_angle_deg))
    denominator = 2.0 * math.cos(theta) ** 2 * (
        distance * math.tan(theta) - float(vertical_difference_m)
    )
    if denominator <= 0.0:
        return None
    return math.sqrt(float(gravity_m_s2) * distance * distance / denominator)


def distance_adaptive_release_angle_deg(
    horizontal_distance_m: float,
    *,
    close_distance_m: float = 1.5,
    close_angle_deg: float = 65.0,
    midrange_distance_m: float = 3.0,
    midrange_angle_deg: float = 55.0,
    three_point_distance_m: float = 6.75,
    three_point_angle_deg: float = 45.0,
) -> float:
    """Interpolate a launch angle through close, midrange and 3PT anchors."""
    distance = max(0.0, float(horizontal_distance_m))
    close_distance = max(0.0, float(close_distance_m))
    midrange_distance = max(close_distance + 1e-6, float(midrange_distance_m))
    three_point_distance = max(
        midrange_distance + 1e-6,
        float(three_point_distance_m),
    )

    if distance <= close_distance:
        return float(close_angle_deg)
    if distance <= midrange_distance:
        fraction = (
            (distance - close_distance)
            / (midrange_distance - close_distance)
        )
        return float(close_angle_deg) + fraction * (
            float(midrange_angle_deg) - float(close_angle_deg)
        )
    if distance <= three_point_distance:
        fraction = (
            (distance - midrange_distance)
            / (three_point_distance - midrange_distance)
        )
        return float(midrange_angle_deg) + fraction * (
            float(three_point_angle_deg) - float(midrange_angle_deg)
        )
    return float(three_point_angle_deg)


def estimate_release_height_m(
    release_y_px: float,
    standing_ankle_y_px: float,
    nose_to_ankle_px: float,
    player_height_m: float,
    *,
    nose_to_ankle_height_fraction: float = 0.92,
    ankle_to_floor_height_fraction: float = 0.04,
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate release height and local scale from a known player height."""
    values = (
        release_y_px,
        standing_ankle_y_px,
        nose_to_ankle_px,
        player_height_m,
        nose_to_ankle_height_fraction,
        ankle_to_floor_height_fraction,
    )
    if not all(math.isfinite(float(value)) for value in values):
        return None, None
    if (
        player_height_m <= 0.0
        or nose_to_ankle_px <= 1e-6
        or nose_to_ankle_height_fraction <= 0.0
    ):
        return None, None

    # MediaPipe measures nose-to-ankle pixels, not full head-to-floor height.
    meters_per_pixel = (
        float(player_height_m) * float(nose_to_ankle_height_fraction)
        / float(nose_to_ankle_px)
    )
    ankle_height_m = (
        float(player_height_m) * float(ankle_to_floor_height_fraction)
    )
    release_height = (
        (float(standing_ankle_y_px) - float(release_y_px))
        * meters_per_pixel
        + ankle_height_m
    )
    return max(0.0, release_height), meters_per_pixel


def pose_nose_to_ankle_length_px(
    image_landmarks: Optional[Dict[str, dict]],
) -> Optional[float]:
    """Measure anatomical chain length without shrinking when knees bend."""
    if not image_landmarks:
        return None

    def point(name: str) -> Optional[Point]:
        landmark = image_landmarks.get(name)
        if landmark is None or float(landmark.get("visibility", 0.0)) < 0.3:
            return None
        return float(landmark["x"]), float(landmark["y"])

    def midpoint(first: Optional[Point], second: Optional[Point]) -> Optional[Point]:
        available = [value for value in (first, second) if value is not None]
        if not available:
            return None
        return (
            sum(value[0] for value in available) / len(available),
            sum(value[1] for value in available) / len(available),
        )

    def distance(first: Optional[Point], second: Optional[Point]) -> Optional[float]:
        if first is None or second is None:
            return None
        return math.hypot(second[0] - first[0], second[1] - first[1])

    nose = point("nose")
    shoulders = midpoint(point("left_shoulder"), point("right_shoulder"))
    hips = midpoint(point("left_hip"), point("right_hip"))
    head_neck = distance(nose, shoulders)
    torso = distance(shoulders, hips)
    if head_neck is None or torso is None:
        return None

    leg_lengths: List[float] = []
    for side in ("left", "right"):
        hip = point(f"{side}_hip")
        knee = point(f"{side}_knee")
        ankle = point(f"{side}_ankle")
        thigh = distance(hip, knee)
        shin = distance(knee, ankle)
        if thigh is not None and shin is not None:
            leg_lengths.append(thigh + shin)
    if not leg_lengths:
        return None

    chain_length = head_neck + torso + float(np.median(leg_lengths))
    return chain_length if math.isfinite(chain_length) and chain_length > 1.0 else None


def ideal_physical_trajectory_pixels(
    release_xy: Point,
    target_xy: Point,
    horizontal_distance_m: float,
    vertical_difference_m: float,
    release_angle_deg: float = 45.0,
    samples: int = 80,
    meters_per_pixel: Optional[float] = None,
) -> List[Point]:
    """Project a metric projectile path between its two image endpoints."""
    release_x, release_y = map(float, release_xy)
    target_x, target_y = map(float, target_xy)
    distance_m = float(horizontal_distance_m)
    vertical_m = float(vertical_difference_m)
    speed = required_release_speed(
        distance_m,
        vertical_m,
        release_angle_deg,
    )
    if speed is None or abs(vertical_m) <= 1e-6:
        return ideal_trajectory_pixels(
            release_xy,
            target_xy,
            release_angle_deg,
            samples,
        )

    theta = math.radians(float(release_angle_deg))
    metric_projection = (
        meters_per_pixel is not None
        and math.isfinite(float(meters_per_pixel))
        and float(meters_per_pixel) > 1e-9
    )
    endpoint_correction_y = 0.0
    if metric_projection:
        raw_target_y = release_y - vertical_m / float(meters_per_pixel)
        endpoint_correction_y = target_y - raw_target_y
    points: List[Point] = []
    for fraction in np.linspace(0.0, 1.0, max(2, int(samples))):
        fraction_f = float(fraction)
        x_m = fraction_f * distance_m
        height_m = (
            x_m * math.tan(theta)
            - 9.81 * x_m * x_m
            / (2.0 * speed * speed * math.cos(theta) ** 2)
        )
        projected_y = (
            release_y
            - height_m / float(meters_per_pixel)
            + fraction_f * endpoint_correction_y
            if metric_projection
            else release_y
            - height_m * (release_y - target_y) / vertical_m
        )
        points.append(
            (
                release_x + (target_x - release_x) * fraction_f,
                projected_y,
            )
        )
    return points


class IdealTrajectoryTracker:
    """Capture one release, build its ideal arc, and compare observed points."""

    def __init__(
        self,
        *,
        release_angle_deg: float = 45.0,
        angle_mode: str = "fixed",
        close_distance_m: float = 1.5,
        close_angle_deg: float = 65.0,
        midrange_distance_m: float = 3.0,
        midrange_angle_deg: float = 55.0,
        three_point_distance_m: float = 6.75,
        three_point_angle_deg: float = 45.0,
        samples: int = 80,
        rim_diameter_m: float = 0.457,
        rim_height_m: float = 3.05,
        minimum_vertical_difference_m: float = 0.25,
        minimum_wrist_vertical_difference_m: float = 0.05,
        velocity_fit_window_s: float = 0.2,
        velocity_fit_min_points: int = 3,
        velocity_fit_min_direct_points: int = 2,
        velocity_fit_outlier_threshold_rim_radii: float = 0.75,
        velocity_min_toward_rim_m_s: float = 0.25,
        velocity_min_upward_m_s: float = 0.25,
        velocity_max_speed_m_s: float = 20.0,
        reachability_vertical_tolerance_m: Optional[float] = None,
        comparison_min_points: int = 3,
    ) -> None:
        self.release_angle_deg = float(release_angle_deg)
        self.angle_mode = str(angle_mode).strip().lower()
        self.close_distance_m = float(close_distance_m)
        self.close_angle_deg = float(close_angle_deg)
        self.midrange_distance_m = float(midrange_distance_m)
        self.midrange_angle_deg = float(midrange_angle_deg)
        self.three_point_distance_m = float(three_point_distance_m)
        self.three_point_angle_deg = float(three_point_angle_deg)
        self.samples = max(2, int(samples))
        self.rim_diameter_m = max(1e-6, float(rim_diameter_m))
        self.rim_height_m = max(1e-6, float(rim_height_m))
        self.minimum_vertical_difference_m = max(
            0.0, float(minimum_vertical_difference_m)
        )
        self.minimum_wrist_vertical_difference_m = max(
            0.0,
            float(minimum_wrist_vertical_difference_m),
        )
        self.velocity_fit_window_s = max(0.01, float(velocity_fit_window_s))
        self.velocity_fit_min_points = max(2, int(velocity_fit_min_points))
        self.velocity_fit_min_direct_points = max(
            1, int(velocity_fit_min_direct_points)
        )
        self.velocity_fit_outlier_threshold_rim_radii = max(
            0.05,
            float(velocity_fit_outlier_threshold_rim_radii),
        )
        self.velocity_min_toward_rim_m_s = max(
            0.0,
            float(velocity_min_toward_rim_m_s),
        )
        self.velocity_min_upward_m_s = max(
            0.0,
            float(velocity_min_upward_m_s),
        )
        self.velocity_max_speed_m_s = max(
            1.0,
            float(velocity_max_speed_m_s),
        )
        self.reachability_vertical_tolerance_m = max(
            0.0,
            float(
                self.rim_diameter_m / 2.0
                if reachability_vertical_tolerance_m is None
                else reachability_vertical_tolerance_m
            ),
        )
        self.comparison_min_points = max(2, int(comparison_min_points))
        self.reset()

    def reset(self) -> None:
        self._release: Optional[_RelativePoint] = None
        self._observed: List[_RelativePoint] = []
        self._crossing_relative_x: Optional[float] = None
        self._crossing_timestamp_ms: Optional[int] = None
        self._target_relative_xy: Point = (0.0, 0.0)
        self._release_height_m: Optional[float] = None
        self._meters_per_pixel_at_release: Optional[float] = None
        self._release_rim_radius_px: Optional[float] = None
        self._horizontal_distance_override_m: Optional[float] = None
        self._kinematics_start_timestamp_ms: Optional[int] = None
        self._release_anchor_source = "ball"
        self._active_release_angle_deg = self.release_angle_deg
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start_from_relative_points(
        self,
        points: Sequence[tuple],
        *,
        target_relative_xy: Point = (0.0, 0.0),
        release_height_m: Optional[float] = None,
        meters_per_pixel_at_release: Optional[float] = None,
        rim_radius_px: Optional[float] = None,
        release_relative_xy: Optional[Point] = None,
        kinematics_start_timestamp_ms: Optional[int] = None,
        horizontal_distance_m_override: Optional[float] = None,
    ) -> None:
        """Start from release-backtracked rim-relative observations."""
        self.reset()
        if not points:
            return
        self._target_relative_xy = (
            float(target_relative_xy[0]),
            float(target_relative_xy[1]),
        )
        if (
            horizontal_distance_m_override is not None
            and math.isfinite(float(horizontal_distance_m_override))
            and float(horizontal_distance_m_override) > 0.0
        ):
            self._horizontal_distance_override_m = float(
                horizontal_distance_m_override
            )
        if (
            release_height_m is not None
            and meters_per_pixel_at_release is not None
            and rim_radius_px is not None
            and math.isfinite(float(release_height_m))
            and math.isfinite(float(meters_per_pixel_at_release))
            and math.isfinite(float(rim_radius_px))
            and 0.0 <= float(release_height_m)
            and self.rim_height_m - float(release_height_m)
            >= (
                self.minimum_wrist_vertical_difference_m
                if release_relative_xy is not None
                else self.minimum_vertical_difference_m
            )
            and float(meters_per_pixel_at_release) > 0.0
            and float(rim_radius_px) > 0.0
        ):
            self._release_height_m = float(release_height_m)
            self._meters_per_pixel_at_release = float(meters_per_pixel_at_release)
            self._release_rim_radius_px = float(rim_radius_px)
        relative = [self._coerce_relative_point(point) for point in points]
        self._release = (
            _RelativePoint(
                float(release_relative_xy[0]),
                float(release_relative_xy[1]),
                relative[0].timestamp_ms,
                "wrist_proxy",
                False,
            )
            if release_relative_xy is not None
            else relative[0]
        )
        self._release_anchor_source = (
            "wrist_proxy" if release_relative_xy is not None else "ball"
        )
        self._observed = relative
        self._kinematics_start_timestamp_ms = (
            int(kinematics_start_timestamp_ms)
            if kinematics_start_timestamp_ms is not None
            else relative[0].timestamp_ms
        )
        physical = self._physical_distances()
        if self.angle_mode == "distance_adaptive" and physical is not None:
            self._active_release_angle_deg = distance_adaptive_release_angle_deg(
                physical[0],
                close_distance_m=self.close_distance_m,
                close_angle_deg=self.close_angle_deg,
                midrange_distance_m=self.midrange_distance_m,
                midrange_angle_deg=self.midrange_angle_deg,
                three_point_distance_m=self.three_point_distance_m,
                three_point_angle_deg=self.three_point_angle_deg,
            )
        self._active = True

    def update(
        self,
        snapshot: Optional[BallSnapshot],
        *,
        released_this_frame: bool,
        shot_finished: bool,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
        crossing_xy: Optional[Point] = None,
        crossing_timestamp_ms: Optional[int] = None,
    ) -> None:
        geometry = self._valid_geometry(rim_center_xy, rim_radius)
        observed = snapshot is not None and snapshot.is_visual_observation

        if released_this_frame:
            self.reset()
            if geometry and observed:
                self._active = True
                relative = self._relative_point(
                    snapshot, rim_center_xy, float(rim_radius)
                )
                self._release = relative
                self._observed.append(relative)
                self._kinematics_start_timestamp_ms = relative.timestamp_ms

        elif self._active and geometry and observed:
            relative = self._relative_point(
                snapshot, rim_center_xy, float(rim_radius)
            )
            if (
                not self._observed
                or relative.timestamp_ms != self._observed[-1].timestamp_ms
            ):
                self._observed.append(relative)

        if (
            self._active
            and geometry
            and crossing_xy is not None
            and self._crossing_relative_x is None
        ):
            self._crossing_relative_x = (
                float(crossing_xy[0]) - float(rim_center_xy[0])
            ) / float(rim_radius)
            self._crossing_timestamp_ms = (
                int(crossing_timestamp_ms)
                if crossing_timestamp_ms is not None
                else snapshot.timestamp_ms if snapshot is not None else None
            )

        if shot_finished:
            self._active = False

    def release_screen_xy(
        self,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
    ) -> Optional[Point]:
        if self._release is None or not self._valid_geometry(
            rim_center_xy, rim_radius
        ):
            return None
        return self._to_screen(
            (self._release.x, self._release.y),
            rim_center_xy,
            float(rim_radius),
        )

    def ideal_screen_curve(
        self,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
    ) -> List[Point]:
        release_xy = self.release_screen_xy(rim_center_xy, rim_radius)
        target_xy = self.target_screen_xy(rim_center_xy, rim_radius)
        if release_xy is None or target_xy is None:
            return []
        physical = self._physical_distances()
        if physical is not None:
            horizontal_m, vertical_m = physical
            return ideal_physical_trajectory_pixels(
                release_xy,
                target_xy,
                horizontal_m,
                vertical_m,
                self._active_release_angle_deg,
                self.samples,
                self._meters_per_pixel_at_release,
            )
        return ideal_trajectory_pixels(
            release_xy,
            target_xy,
            self._active_release_angle_deg,
            self.samples,
        )

    def target_screen_xy(
        self,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
    ) -> Optional[Point]:
        """Return the ideal rim target using the current stabilized geometry."""
        if self._release is None or not self._valid_geometry(
            rim_center_xy, rim_radius
        ):
            return None
        return self._to_screen(
            self._target_relative_xy,
            rim_center_xy,
            float(rim_radius),
        )

    def comparison(self) -> Optional[TrajectoryComparison]:
        release = self._release
        if release is None:
            return None

        target_x, target_y = self._target_relative_xy
        physical = self._physical_distances()
        calibrated = physical is not None
        if physical is not None:
            horizontal_m, vertical_m = physical
            velocity_scale_m = (
                float(self._release_rim_radius_px)
                * float(self._meters_per_pixel_at_release)
            )
        else:
            rim_radius_m = self.rim_diameter_m / 2.0
            horizontal_radii = abs(target_x - release.x)
            vertical_radii = release.y - target_y
            horizontal_m = (
                self._horizontal_distance_override_m
                if self._horizontal_distance_override_m is not None
                else horizontal_radii * rim_radius_m
            )
            vertical_m = vertical_radii * rim_radius_m
            velocity_scale_m = rim_radius_m
        horizontal_distance_source = (
            "fiba_court_location"
            if self._horizontal_distance_override_m is not None
            else "player_height_image"
            if calibrated
            else "rim_diameter_fallback"
        )
        straight_m = math.hypot(horizontal_m, vertical_m)
        target_speed = required_release_speed(
            horizontal_m,
            vertical_m,
            self._active_release_angle_deg,
        )

        (
            release_to_rim_time_s,
            release_to_rim_displacement_m,
            release_to_rim_path_length_m,
            release_to_rim_average_speed_m_s,
        ) = self._release_to_rim_average_speed(velocity_scale_m)
        (
            observed_angle,
            observed_speed,
            early_kinematics_status,
        ) = self._observed_release_kinematics(velocity_scale_m)
        early_reachability_status = "not_evaluated"
        predicted_height_at_rim_distance_m = None
        reachability_height_margin_m = None
        minimum_reachable_speed_m_s = None
        if observed_angle is not None and observed_speed is not None:
            (
                early_reachability_status,
                predicted_height_at_rim_distance_m,
                reachability_height_margin_m,
                minimum_reachable_speed_m_s,
            ) = self._launch_reachability(
                observed_angle,
                observed_speed,
                horizontal_m,
                vertical_m,
                distance_is_calibrated=(
                    calibrated
                    or self._horizontal_distance_override_m is not None
                ),
            )
            if early_reachability_status == "below_required_height":
                observed_angle = None
                observed_speed = None
                early_kinematics_status = "physically_unreachable_target"
        observed_kinematics_source = "early_post_release_fit"
        if observed_angle is None or observed_speed is None:
            flight_kinematics = self._release_to_rim_ballistic_kinematics(
                vertical_m,
                release_to_rim_time_s,
                velocity_scale_m,
            )
            if flight_kinematics is not None:
                observed_angle, observed_speed = flight_kinematics
                observed_kinematics_source = "release_to_rim_ballistic"
            else:
                observed_kinematics_source = "unavailable"
        angle_error = (
            observed_angle - self._active_release_angle_deg
            if observed_angle is not None
            else None
        )
        speed_error = (
            observed_speed - target_speed
            if observed_speed is not None and target_speed is not None
            else None
        )

        path_rmse = self._path_comparison()
        apex = self._apex_comparison(horizontal_m, target_speed)
        apex_error = (
            apex.observed_height_rim_radii - apex.ideal_height_rim_radii
            if apex.observed_height_rim_radii is not None
            and apex.ideal_height_rim_radii is not None
            else None
        )
        apex_time_error = (
            apex.observed_time_from_release_s
            - apex.ideal_time_from_release_s
            if apex.observed_time_from_release_s is not None
            and apex.ideal_time_from_release_s is not None
            else None
        )
        apex_position_error_progress = (
            apex.observed_progress - apex.ideal_progress
            if apex.observed_progress is not None
            and apex.ideal_progress is not None
            else None
        )
        apex_position_error_m = (
            apex.observed_distance_m - apex.ideal_distance_m
            if apex.observed_distance_m is not None
            and apex.ideal_distance_m is not None
            else None
        )
        (
            target_entry_angle,
            observed_entry_angle,
            entry_angle_fit_status,
        ) = self._entry_angle_comparison(horizontal_m, vertical_m)
        entry_angle_error = (
            observed_entry_angle - target_entry_angle
            if observed_entry_angle is not None
            and target_entry_angle is not None
            else None
        )

        return TrajectoryComparison(
            target_release_angle_deg=self._active_release_angle_deg,
            observed_release_angle_deg=observed_angle,
            release_angle_error_deg=angle_error,
            target_release_speed_m_s=target_speed,
            observed_release_speed_m_s=observed_speed,
            release_speed_error_m_s=speed_error,
            target_entry_angle_deg=target_entry_angle,
            observed_entry_angle_deg=observed_entry_angle,
            entry_angle_error_deg=entry_angle_error,
            entry_angle_fit_status=entry_angle_fit_status,
            observed_kinematics_source=observed_kinematics_source,
            early_kinematics_status=early_kinematics_status,
            early_reachability_status=early_reachability_status,
            predicted_height_at_rim_distance_m=(
                predicted_height_at_rim_distance_m
            ),
            reachability_height_margin_m=reachability_height_margin_m,
            minimum_reachable_speed_m_s=minimum_reachable_speed_m_s,
            reachability_vertical_tolerance_m=(
                self.reachability_vertical_tolerance_m
            ),
            release_height_m=self._release_height_m if calibrated else None,
            rim_height_m=self.rim_height_m,
            horizontal_distance_m=horizontal_m,
            horizontal_distance_source=horizontal_distance_source,
            vertical_distance_m=vertical_m,
            straight_line_distance_m=straight_m,
            release_to_rim_time_s=release_to_rim_time_s,
            release_to_rim_displacement_m=release_to_rim_displacement_m,
            release_to_rim_path_length_m=release_to_rim_path_length_m,
            release_to_rim_average_speed_m_s=(
                release_to_rim_average_speed_m_s
            ),
            meters_per_pixel_at_release=(
                self._meters_per_pixel_at_release if calibrated else None
            ),
            velocity_calibration=(
                "player_height" if calibrated else "rim_diameter_fallback"
            ),
            release_anchor_source=self._release_anchor_source,
            path_rmse_rim_radii=path_rmse,
            observed_apex_height_rim_radii=(
                apex.observed_height_rim_radii
            ),
            ideal_apex_height_rim_radii=apex.ideal_height_rim_radii,
            apex_height_error_rim_radii=apex_error,
            apex_fit_status=apex.fit_status,
            observed_apex_timestamp_ms=apex.observed_timestamp_ms,
            ideal_apex_timestamp_ms=apex.ideal_timestamp_ms,
            observed_apex_time_from_release_s=(
                apex.observed_time_from_release_s
            ),
            ideal_apex_time_from_release_s=apex.ideal_time_from_release_s,
            apex_time_error_s=apex_time_error,
            observed_apex_x_rim_radii=apex.observed_x_rim_radii,
            ideal_apex_x_rim_radii=apex.ideal_x_rim_radii,
            observed_apex_progress=apex.observed_progress,
            ideal_apex_progress=apex.ideal_progress,
            apex_position_error_progress=apex_position_error_progress,
            observed_apex_distance_m=apex.observed_distance_m,
            ideal_apex_distance_m=apex.ideal_distance_m,
            apex_position_error_m=apex_position_error_m,
            rim_crossing_error_rim_radii=(
                self._crossing_relative_x - target_x
                if self._crossing_relative_x is not None
                else None
            ),
            observed_point_count=len(self._observed),
            observed_direct_point_count=sum(
                point.is_direct_observation for point in self._observed
            ),
            observed_tracked_point_count=sum(
                point.measurement_source == "nanotrack"
                for point in self._observed
            ),
        )

    def _observed_release_kinematics(
        self, velocity_scale_m: float
    ) -> Tuple[Optional[float], Optional[float], str]:
        if not self._observed:
            return None, None, "no_observed_points"

        start_timestamp_ms = (
            self._kinematics_start_timestamp_ms
            if self._kinematics_start_timestamp_ms is not None
            else self._observed[0].timestamp_ms
        )
        eligible = [
            point
            for point in self._observed
            if point.timestamp_ms >= start_timestamp_ms
        ]
        window_ms = int(round(self.velocity_fit_window_s * 1000.0))
        window_end_ms = start_timestamp_ms + window_ms
        if eligible[-1].timestamp_ms < window_end_ms:
            return None, None, "incomplete_post_release_time_window"
        points = [
            point for point in eligible if point.timestamp_ms <= window_end_ms
        ]
        if len(points) < self.velocity_fit_min_points:
            return None, None, "insufficient_post_release_points"
        direct_count = sum(point.is_direct_observation for point in points)
        if direct_count < self.velocity_fit_min_direct_points:
            return None, None, "insufficient_direct_detections"

        times = np.asarray(
            [(point.timestamp_ms - points[0].timestamp_ms) / 1000.0 for point in points],
            dtype=np.float64,
        )
        if times[-1] <= 0.0 or len(np.unique(times)) < 2:
            return None, None, "invalid_timestamps"

        x_values = np.asarray([point.x for point in points], dtype=np.float64)
        y_values = np.asarray([point.y for point in points], dtype=np.float64)
        inliers = np.ones(len(points), dtype=bool)
        minimum_inliers = max(2, int(math.ceil(len(points) * 0.6)))

        # A linear horizontal fit and quadratic vertical fit estimate velocity
        # at the first post-separation point. Iterative rejection prevents one
        # NanoTrack/YOLO correction from dominating that short launch window.
        for _ in range(3):
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < minimum_inliers:
                return None, None, "too_few_fit_inliers"
            y_degree = 2 if inlier_count >= 3 else 1
            x_coefficients = np.polyfit(times[inliers], x_values[inliers], 1)
            y_coefficients = np.polyfit(
                times[inliers],
                y_values[inliers],
                y_degree,
            )
            predicted_x = np.polyval(x_coefficients, times)
            predicted_y = np.polyval(y_coefficients, times)
            residuals = np.hypot(
                x_values - predicted_x,
                y_values - predicted_y,
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            robust_sigma = max(1.4826 * mad, 0.01)
            threshold = max(
                self.velocity_fit_outlier_threshold_rim_radii,
                median + 3.5 * robust_sigma,
            )
            updated_inliers = residuals <= threshold
            if int(np.count_nonzero(updated_inliers)) < minimum_inliers:
                return None, None, "outlier_rejection_failed"
            if np.array_equal(updated_inliers, inliers):
                break
            inliers = updated_inliers

        inlier_count = int(np.count_nonzero(inliers))
        y_degree = 2 if inlier_count >= 3 else 1
        x_coefficients = np.polyfit(times[inliers], x_values[inliers], 1)
        y_coefficients = np.polyfit(times[inliers], y_values[inliers], y_degree)
        vx_radii_s = float(x_coefficients[0])
        vy_radii_s = float(
            y_coefficients[-2] if y_degree == 2 else y_coefficients[0]
        )
        target_x = self._target_relative_xy[0]
        direction_to_rim = 1.0 if target_x >= self._release.x else -1.0
        velocity_toward_rim_m_s = (
            direction_to_rim * vx_radii_s * velocity_scale_m
        )
        velocity_upward_m_s = -vy_radii_s * velocity_scale_m
        observed_speed = math.hypot(
            velocity_toward_rim_m_s,
            velocity_upward_m_s,
        )
        if velocity_toward_rim_m_s < self.velocity_min_toward_rim_m_s:
            return None, None, "not_moving_toward_rim"
        if velocity_upward_m_s < self.velocity_min_upward_m_s:
            return None, None, "not_moving_upward"
        if observed_speed > self.velocity_max_speed_m_s:
            return None, None, "speed_above_plausible_limit"

        observed_angle = math.degrees(
            math.atan2(velocity_upward_m_s, velocity_toward_rim_m_s)
        )
        return observed_angle, observed_speed, "accepted"

    def _launch_reachability(
        self,
        release_angle_deg: float,
        release_speed_m_s: float,
        horizontal_distance_m: float,
        vertical_difference_m: float,
        *,
        distance_is_calibrated: bool,
    ) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
        """Check whether an early launch estimate can reach the rim's height.

        This is a lower-bound gate, not make/miss prediction. A launch may pass
        above the target and still be physically reachable; it is rejected only
        when gravity places it more than one configured tolerance below the rim
        by the time it has covered the known horizontal distance.
        """
        if not distance_is_calibrated:
            return "not_calibrated", None, None, None

        angle_rad = math.radians(float(release_angle_deg))
        speed_m_s = float(release_speed_m_s)
        horizontal_m = float(horizontal_distance_m)
        vertical_m = float(vertical_difference_m)
        horizontal_speed_m_s = speed_m_s * math.cos(angle_rad)
        if (
            speed_m_s <= 0.0
            or horizontal_m <= 0.0
            or horizontal_speed_m_s <= 1e-6
        ):
            return "invalid_launch", None, None, None

        flight_time_s = horizontal_m / horizontal_speed_m_s
        predicted_height_m = (
            speed_m_s * math.sin(angle_rad) * flight_time_s
            - 0.5 * 9.81 * flight_time_s * flight_time_s
        )
        height_margin_m = predicted_height_m - vertical_m
        minimum_speed_m_s = required_release_speed(
            horizontal_m,
            vertical_m - self.reachability_vertical_tolerance_m,
            release_angle_deg,
        )
        status = (
            "accepted"
            if height_margin_m >= -self.reachability_vertical_tolerance_m
            else "below_required_height"
        )
        return (
            status,
            predicted_height_m,
            height_margin_m,
            minimum_speed_m_s,
        )

    def _release_to_rim_ballistic_kinematics(
        self,
        vertical_difference_m: float,
        flight_time_s: Optional[float],
        velocity_scale_m: float,
    ) -> Optional[Tuple[float, float]]:
        """Reconstruct launch velocity from release and rim-crossing events."""
        if (
            flight_time_s is None
            or flight_time_s <= 1e-6
            or self._release is None
            or self._crossing_relative_x is None
        ):
            return None

        horizontal_m = (
            self._horizontal_distance_override_m
            if self._horizontal_distance_override_m is not None
            else abs(self._crossing_relative_x - self._release.x)
            * velocity_scale_m
        )
        if horizontal_m <= 1e-6:
            return None
        vx_m_s = horizontal_m / flight_time_s
        vy_m_s = (
            float(vertical_difference_m)
            + 0.5 * 9.81 * flight_time_s * flight_time_s
        ) / flight_time_s
        speed_m_s = math.hypot(vx_m_s, vy_m_s)
        if (
            vx_m_s < self.velocity_min_toward_rim_m_s
            or vy_m_s < self.velocity_min_upward_m_s
            or speed_m_s > self.velocity_max_speed_m_s
        ):
            return None
        return math.degrees(math.atan2(vy_m_s, vx_m_s)), speed_m_s

    def _release_to_rim_average_speed(
        self,
        velocity_scale_m: float,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """Flight time, displacement, path length, and mean path speed."""
        if (
            self._crossing_relative_x is None
            or self._crossing_timestamp_ms is None
            or not self._observed
        ):
            return None, None, None, None

        start_timestamp_ms = (
            self._kinematics_start_timestamp_ms
            if self._kinematics_start_timestamp_ms is not None
            else self._observed[0].timestamp_ms
        )
        start = next(
            (
                point
                for point in self._observed
                if point.timestamp_ms >= start_timestamp_ms
            ),
            None,
        )
        if start is None:
            return None, None, None, None

        elapsed_s = (
            self._crossing_timestamp_ms - start.timestamp_ms
        ) / 1000.0
        if elapsed_s <= 1e-6:
            return None, None, None, None

        displacement_radii = math.hypot(
            self._crossing_relative_x - start.x,
            -start.y,
        )
        displacement_m = displacement_radii * velocity_scale_m
        flight_points = [
            point
            for point in self._observed
            if start.timestamp_ms
            <= point.timestamp_ms
            <= self._crossing_timestamp_ms
        ]
        path_coordinates = [(point.x, point.y) for point in flight_points]
        crossing_point = (self._crossing_relative_x, 0.0)
        if not path_coordinates or path_coordinates[-1] != crossing_point:
            path_coordinates.append(crossing_point)
        path_radii = sum(
            math.hypot(x2 - x1, y2 - y1)
            for (x1, y1), (x2, y2) in zip(
                path_coordinates,
                path_coordinates[1:],
            )
        )
        path_length_m = path_radii * velocity_scale_m
        return (
            elapsed_s,
            displacement_m,
            path_length_m,
            path_length_m / elapsed_s,
        )

    def _path_comparison(
        self,
    ) -> Optional[float]:
        release = self._release
        target_x, target_y = self._target_relative_xy
        if release is None or abs(target_x - release.x) <= 1e-6:
            return None

        direction = 1.0 if target_x >= release.x else -1.0
        distance = abs(target_x - release.x)
        height_to_target = release.y - target_y
        theta = math.radians(self._active_release_angle_deg)
        physical = self._physical_distances()
        curvature = (
            distance * math.tan(theta) - height_to_target
        ) / (distance * distance)

        residuals: List[float] = []
        for point in self._observed:
            progress = direction * (point.x - release.x)
            if progress < 0.0 or progress > distance:
                continue
            ideal_y = self._ideal_relative_y(
                progress,
                distance,
                release,
                target_y,
                theta,
                curvature,
                physical,
            )
            residuals.append(point.y - ideal_y)

        if len(residuals) < self.comparison_min_points:
            return None
        return math.sqrt(float(np.mean(np.square(residuals))))

    def _apex_comparison(
        self,
        horizontal_distance_m: float,
        target_release_speed_m_s: Optional[float],
    ) -> _ApexComparison:
        """Locate observed and ideal peaks in time and along the shot path.

        The observed peak comes from a robust x(t)/y(t) fit rather than the
        single smallest image-y measurement. This suppresses detector jitter.
        Position is expressed as progress from release (0) to rim target (1),
        making it comparable across resolutions and moving-camera footage.
        """
        release = self._release
        target_x, target_y = self._target_relative_xy
        if release is None or abs(target_x - release.x) <= 1e-6:
            return _ApexComparison(
                "invalid_geometry",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        direction = 1.0 if target_x >= release.x else -1.0
        distance = abs(target_x - release.x)
        theta = math.radians(self._active_release_angle_deg)
        height_to_target = release.y - target_y
        curvature = (
            distance * math.tan(theta) - height_to_target
        ) / (distance * distance)
        physical = self._physical_distances()

        # The ideal curve is quadratic in horizontal progress. Evaluating at
        # three exact positions recovers that quadratic and its vertex without
        # introducing sampling-resolution error.
        ideal_progress_samples = np.asarray(
            [0.0, distance / 2.0, distance], dtype=np.float64
        )
        ideal_y_samples = np.asarray(
            [
                self._ideal_relative_y(
                    float(progress),
                    distance,
                    release,
                    target_y,
                    theta,
                    curvature,
                    physical,
                )
                for progress in ideal_progress_samples
            ],
            dtype=np.float64,
        )
        ideal_coefficients = np.polyfit(
            ideal_progress_samples, ideal_y_samples, 2
        )
        ideal_quadratic = float(ideal_coefficients[0])
        if ideal_quadratic > 1e-9:
            ideal_progress_radii = float(
                np.clip(
                    -ideal_coefficients[1] / (2.0 * ideal_quadratic),
                    0.0,
                    distance,
                )
            )
        else:
            ideal_progress_radii = float(
                ideal_progress_samples[int(np.argmin(ideal_y_samples))]
            )
        ideal_y = float(np.polyval(ideal_coefficients, ideal_progress_radii))
        ideal_progress = ideal_progress_radii / distance
        ideal_x = release.x + direction * ideal_progress_radii
        ideal_distance_m = ideal_progress * horizontal_distance_m
        ideal_time_s = None
        if target_release_speed_m_s is not None:
            ideal_horizontal_speed = (
                target_release_speed_m_s * math.cos(theta)
            )
            if ideal_horizontal_speed > 1e-9:
                ideal_time_s = ideal_distance_m / ideal_horizontal_speed

        release_timestamp_ms = (
            self._kinematics_start_timestamp_ms
            if self._kinematics_start_timestamp_ms is not None
            else release.timestamp_ms
        )
        ideal_timestamp_ms = (
            release_timestamp_ms + int(round(ideal_time_s * 1000.0))
            if ideal_time_s is not None
            else None
        )

        fit_status, observed_timestamp_ms, observed_x, observed_y = (
            self._fit_observed_apex(release_timestamp_ms)
        )
        if observed_x is None or observed_y is None:
            return _ApexComparison(
                fit_status,
                None,
                target_y - ideal_y,
                None,
                ideal_timestamp_ms,
                None,
                ideal_time_s,
                None,
                ideal_x,
                None,
                ideal_progress,
                None,
                ideal_distance_m,
            )

        observed_progress_radii = direction * (observed_x - release.x)
        observed_progress = observed_progress_radii / distance
        observed_distance_m = observed_progress * horizontal_distance_m
        observed_time_s = (
            (observed_timestamp_ms - release_timestamp_ms) / 1000.0
            if observed_timestamp_ms is not None
            else None
        )
        return _ApexComparison(
            fit_status,
            target_y - observed_y,
            target_y - ideal_y,
            observed_timestamp_ms,
            ideal_timestamp_ms,
            observed_time_s,
            ideal_time_s,
            observed_x,
            ideal_x,
            observed_progress,
            ideal_progress,
            observed_distance_m,
            ideal_distance_m,
        )

    def _fit_observed_apex(
        self,
        release_timestamp_ms: int,
    ) -> Tuple[str, Optional[int], Optional[float], Optional[float]]:
        """Fit the measured flight and return its vertex in rim units."""
        flight_fit = self._fit_observed_flight(release_timestamp_ms)
        if (
            flight_fit.status != "accepted"
            or flight_fit.x_coefficients is None
            or flight_fit.y_coefficients is None
            or flight_fit.first_time_s is None
            or flight_fit.last_time_s is None
        ):
            return flight_fit.status, None, None, None

        x_coefficients = np.asarray(
            flight_fit.x_coefficients, dtype=np.float64
        )
        y_coefficients = np.asarray(
            flight_fit.y_coefficients, dtype=np.float64
        )
        quadratic = float(y_coefficients[0])
        if quadratic <= 1e-9:
            return "non_ballistic_curve", None, None, None
        apex_time_s = -float(y_coefficients[1]) / (2.0 * quadratic)
        if (
            apex_time_s < flight_fit.first_time_s
            or apex_time_s > flight_fit.last_time_s
        ):
            return "apex_outside_observed_flight", None, None, None

        apex_x = float(np.polyval(x_coefficients, apex_time_s))
        apex_y = float(np.polyval(y_coefficients, apex_time_s))
        apex_timestamp_ms = release_timestamp_ms + int(
            round(apex_time_s * 1000.0)
        )
        return "accepted", apex_timestamp_ms, apex_x, apex_y

    def _fit_observed_flight(
        self,
        release_timestamp_ms: int,
    ) -> _ObservedFlightFit:
        """Robustly fit all visual measurements from release to crossing."""
        end_timestamp_ms = (
            self._crossing_timestamp_ms
            if self._crossing_timestamp_ms is not None
            else self._observed[-1].timestamp_ms if self._observed else None
        )
        points = [
            point
            for point in self._observed
            if point.timestamp_ms >= release_timestamp_ms
            and (
                end_timestamp_ms is None
                or point.timestamp_ms <= end_timestamp_ms
            )
        ]
        minimum_points = max(5, self.comparison_min_points)
        if len(points) < minimum_points:
            return _ObservedFlightFit(
                "insufficient_points", release_timestamp_ms
            )

        times = np.asarray(
            [
                (point.timestamp_ms - release_timestamp_ms) / 1000.0
                for point in points
            ],
            dtype=np.float64,
        )
        if len(np.unique(times)) < 3 or times[-1] <= times[0]:
            return _ObservedFlightFit(
                "invalid_timestamps", release_timestamp_ms
            )

        x_values = np.asarray([point.x for point in points], dtype=np.float64)
        y_values = np.asarray([point.y for point in points], dtype=np.float64)
        inliers = np.ones(len(points), dtype=bool)
        minimum_inliers = max(
            minimum_points,
            int(math.ceil(len(points) * 0.6)),
        )

        for _ in range(3):
            if int(np.count_nonzero(inliers)) < minimum_inliers:
                return _ObservedFlightFit(
                    "too_few_fit_inliers", release_timestamp_ms
                )
            x_coefficients = np.polyfit(times[inliers], x_values[inliers], 1)
            y_coefficients = np.polyfit(times[inliers], y_values[inliers], 2)
            predicted_x = np.polyval(x_coefficients, times)
            predicted_y = np.polyval(y_coefficients, times)
            residuals = np.hypot(
                x_values - predicted_x,
                y_values - predicted_y,
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            robust_sigma = max(1.4826 * mad, 0.01)
            threshold = max(
                self.velocity_fit_outlier_threshold_rim_radii,
                median + 3.5 * robust_sigma,
            )
            updated_inliers = residuals <= threshold
            if int(np.count_nonzero(updated_inliers)) < minimum_inliers:
                return _ObservedFlightFit(
                    "outlier_rejection_failed", release_timestamp_ms
                )
            if np.array_equal(updated_inliers, inliers):
                break
            inliers = updated_inliers

        x_coefficients = np.polyfit(times[inliers], x_values[inliers], 1)
        y_coefficients = np.polyfit(times[inliers], y_values[inliers], 2)
        first_time_s = float(np.min(times[inliers]))
        last_time_s = float(np.max(times[inliers]))
        return _ObservedFlightFit(
            status="accepted",
            release_timestamp_ms=release_timestamp_ms,
            first_time_s=first_time_s,
            last_time_s=last_time_s,
            x_coefficients=(
                float(x_coefficients[0]),
                float(x_coefficients[1]),
            ),
            y_coefficients=(
                float(y_coefficients[0]),
                float(y_coefficients[1]),
                float(y_coefficients[2]),
            ),
        )

    def _entry_angle_comparison(
        self,
        horizontal_distance_m: float,
        vertical_distance_m: float,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Return ideal/observed downward angles at the rim plane."""
        horizontal_m = float(horizontal_distance_m)
        if horizontal_m <= 1e-9:
            return None, None, "invalid_geometry"

        theta = math.radians(self._active_release_angle_deg)
        # For a projectile constrained to pass through (d, delta_h), its
        # downward slope at the target is tan(theta) - 2*delta_h/d.
        target_downward_slope = (
            math.tan(theta)
            - 2.0 * float(vertical_distance_m) / horizontal_m
        )
        target_entry_angle = (
            math.degrees(math.atan(target_downward_slope))
            if target_downward_slope > 0.0
            else None
        )

        if self._crossing_timestamp_ms is None:
            return target_entry_angle, None, "rim_crossing_unavailable"

        release_timestamp_ms = (
            self._kinematics_start_timestamp_ms
            if self._kinematics_start_timestamp_ms is not None
            else self._release.timestamp_ms if self._release is not None else 0
        )
        flight_fit = self._fit_observed_flight(release_timestamp_ms)
        if (
            flight_fit.status != "accepted"
            or flight_fit.x_coefficients is None
            or flight_fit.y_coefficients is None
        ):
            return target_entry_angle, None, flight_fit.status

        crossing_time_s = (
            self._crossing_timestamp_ms - release_timestamp_ms
        ) / 1000.0
        if crossing_time_s < 0.0:
            return target_entry_angle, None, "crossing_before_release"

        horizontal_velocity = abs(float(flight_fit.x_coefficients[0]))
        quadratic, linear, _ = flight_fit.y_coefficients
        downward_velocity = 2.0 * quadratic * crossing_time_s + linear
        if horizontal_velocity <= 1e-9:
            return target_entry_angle, None, "no_horizontal_velocity"
        if downward_velocity <= 0.0:
            return target_entry_angle, None, "not_descending_at_rim"

        observed_entry_angle = math.degrees(
            math.atan2(downward_velocity, horizontal_velocity)
        )
        return target_entry_angle, observed_entry_angle, "accepted"

    def _physical_distances(self) -> Optional[Tuple[float, float]]:
        if (
            self._release is None
            or self._release_height_m is None
            or self._meters_per_pixel_at_release is None
            or self._release_rim_radius_px is None
        ):
            return None
        if self._horizontal_distance_override_m is not None:
            horizontal_m = self._horizontal_distance_override_m
        else:
            horizontal_pixels = (
                abs(self._target_relative_xy[0] - self._release.x)
                * self._release_rim_radius_px
            )
            horizontal_m = horizontal_pixels * self._meters_per_pixel_at_release
        vertical_m = self.rim_height_m - self._release_height_m
        minimum_vertical_m = (
            self.minimum_wrist_vertical_difference_m
            if self._release_anchor_source == "wrist_proxy"
            else self.minimum_vertical_difference_m
        )
        if (
            horizontal_m <= 1e-6
            or vertical_m < minimum_vertical_m
        ):
            return None
        return horizontal_m, vertical_m

    def _ideal_relative_y(
        self,
        progress_radii: float,
        distance_radii: float,
        release: _RelativePoint,
        target_y: float,
        theta: float,
        fallback_curvature: float,
        physical: Optional[Tuple[float, float]],
    ) -> float:
        if physical is None:
            ideal_height = (
                progress_radii * math.tan(theta)
                - fallback_curvature * progress_radii * progress_radii
            )
            return release.y - ideal_height

        horizontal_m, vertical_m = physical
        speed = required_release_speed(
            horizontal_m,
            vertical_m,
            self._active_release_angle_deg,
        )
        if speed is None or vertical_m <= 1e-6:
            return release.y
        x_m = progress_radii / distance_radii * horizontal_m
        height_m = (
            x_m * math.tan(theta)
            - 9.81 * x_m * x_m
            / (2.0 * speed * speed * math.cos(theta) ** 2)
        )
        metric_scale_m_per_radius = (
            float(self._release_rim_radius_px)
            * float(self._meters_per_pixel_at_release)
            if self._release_rim_radius_px is not None
            and self._meters_per_pixel_at_release is not None
            else None
        )
        if metric_scale_m_per_radius is None or metric_scale_m_per_radius <= 1e-9:
            return release.y - height_m * (release.y - target_y) / vertical_m

        fraction = progress_radii / distance_radii
        raw_target_y = release.y - vertical_m / metric_scale_m_per_radius
        endpoint_correction_y = target_y - raw_target_y
        return (
            release.y
            - height_m / metric_scale_m_per_radius
            + fraction * endpoint_correction_y
        )

    @staticmethod
    def _valid_geometry(
        rim_center_xy: Optional[Point], rim_radius: Optional[float]
    ) -> bool:
        return (
            rim_center_xy is not None
            and rim_radius is not None
            and math.isfinite(float(rim_radius))
            and float(rim_radius) > 1e-6
        )

    @staticmethod
    def _relative_point(
        snapshot: BallSnapshot, rim_center_xy: Point, rim_radius: float
    ) -> _RelativePoint:
        return _RelativePoint(
            x=(snapshot.x - float(rim_center_xy[0])) / rim_radius,
            y=(snapshot.y - float(rim_center_xy[1])) / rim_radius,
            timestamp_ms=snapshot.timestamp_ms,
            measurement_source=snapshot.measurement_source,
            is_direct_observation=snapshot.is_direct_observation,
        )

    @staticmethod
    def _coerce_relative_point(point: tuple) -> _RelativePoint:
        if len(point) < 3:
            raise ValueError("A trajectory point requires x, y, and timestamp_ms")
        source = str(point[3]) if len(point) >= 4 else "unknown"
        direct = bool(point[4]) if len(point) >= 5 else source in {
            "unknown",
            "yolo",
            "color",
        }
        return _RelativePoint(
            float(point[0]),
            float(point[1]),
            int(point[2]),
            source,
            direct,
        )

    @staticmethod
    def _to_screen(
        relative_xy: Point, rim_center_xy: Point, rim_radius: float
    ) -> Point:
        return (
            float(rim_center_xy[0]) + float(relative_xy[0]) * rim_radius,
            float(rim_center_xy[1]) + float(relative_xy[1]) * rim_radius,
        )
