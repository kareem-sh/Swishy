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
    release_height_m: Optional[float]
    rim_height_m: Optional[float]
    horizontal_distance_m: Optional[float]
    vertical_distance_m: Optional[float]
    straight_line_distance_m: Optional[float]
    meters_per_pixel_at_release: Optional[float]
    velocity_calibration: str
    release_anchor_source: str
    path_rmse_rim_radii: Optional[float]
    observed_apex_height_rim_radii: Optional[float]
    ideal_apex_height_rim_radii: Optional[float]
    apex_height_error_rim_radii: Optional[float]
    rim_crossing_error_rim_radii: Optional[float]
    observed_point_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _RelativePoint:
    x: float
    y: float
    timestamp_ms: int


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
    points: List[Point] = []
    for fraction in np.linspace(0.0, 1.0, max(2, int(samples))):
        fraction_f = float(fraction)
        x_m = fraction_f * distance_m
        height_m = (
            x_m * math.tan(theta)
            - 9.81 * x_m * x_m
            / (2.0 * speed * speed * math.cos(theta) ** 2)
        )
        points.append(
            (
                release_x + (target_x - release_x) * fraction_f,
                release_y
                - height_m * (release_y - target_y) / vertical_m,
            )
        )
    return points


class IdealTrajectoryTracker:
    """Capture one release, build its ideal arc, and compare observed points."""

    def __init__(
        self,
        *,
        release_angle_deg: float = 45.0,
        samples: int = 80,
        rim_diameter_m: float = 0.457,
        rim_height_m: float = 3.05,
        minimum_vertical_difference_m: float = 0.25,
        velocity_fit_points: int = 5,
        comparison_min_points: int = 3,
    ) -> None:
        self.release_angle_deg = float(release_angle_deg)
        self.samples = max(2, int(samples))
        self.rim_diameter_m = max(1e-6, float(rim_diameter_m))
        self.rim_height_m = max(1e-6, float(rim_height_m))
        self.minimum_vertical_difference_m = max(
            0.0, float(minimum_vertical_difference_m)
        )
        self.velocity_fit_points = max(2, int(velocity_fit_points))
        self.comparison_min_points = max(2, int(comparison_min_points))
        self.reset()

    def reset(self) -> None:
        self._release: Optional[_RelativePoint] = None
        self._observed: List[_RelativePoint] = []
        self._crossing_relative_x: Optional[float] = None
        self._target_relative_xy: Point = (0.0, 0.0)
        self._release_height_m: Optional[float] = None
        self._meters_per_pixel_at_release: Optional[float] = None
        self._release_rim_radius_px: Optional[float] = None
        self._release_anchor_source = "ball"
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start_from_relative_points(
        self,
        points: Sequence[Tuple[float, float, int]],
        *,
        target_relative_xy: Point = (0.0, 0.0),
        release_height_m: Optional[float] = None,
        meters_per_pixel_at_release: Optional[float] = None,
        rim_radius_px: Optional[float] = None,
        release_relative_xy: Optional[Point] = None,
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
            release_height_m is not None
            and meters_per_pixel_at_release is not None
            and rim_radius_px is not None
            and math.isfinite(float(release_height_m))
            and math.isfinite(float(meters_per_pixel_at_release))
            and math.isfinite(float(rim_radius_px))
            and 0.0 <= float(release_height_m)
            and self.rim_height_m - float(release_height_m)
            >= self.minimum_vertical_difference_m
            and float(meters_per_pixel_at_release) > 0.0
            and float(rim_radius_px) > 0.0
        ):
            self._release_height_m = float(release_height_m)
            self._meters_per_pixel_at_release = float(meters_per_pixel_at_release)
            self._release_rim_radius_px = float(rim_radius_px)
        relative = [
            _RelativePoint(float(x), float(y), int(timestamp_ms))
            for x, y, timestamp_ms in points
        ]
        self._release = (
            _RelativePoint(
                float(release_relative_xy[0]),
                float(release_relative_xy[1]),
                relative[0].timestamp_ms,
            )
            if release_relative_xy is not None
            else relative[0]
        )
        self._release_anchor_source = (
            "wrist_proxy" if release_relative_xy is not None else "ball"
        )
        self._observed = relative
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
    ) -> None:
        geometry = self._valid_geometry(rim_center_xy, rim_radius)
        observed = snapshot is not None and not snapshot.is_interpolated

        if released_this_frame:
            self.reset()
            if geometry and observed:
                self._active = True
                relative = self._relative_point(
                    snapshot, rim_center_xy, float(rim_radius)
                )
                self._release = relative
                self._observed.append(relative)

        elif self._active and geometry and observed:
            relative = self._relative_point(
                snapshot, rim_center_xy, float(rim_radius)
            )
            if (
                not self._observed
                or relative.timestamp_ms != self._observed[-1].timestamp_ms
            ):
                self._observed.append(relative)

        if self._active and geometry and crossing_xy is not None:
            self._crossing_relative_x = (
                float(crossing_xy[0]) - float(rim_center_xy[0])
            ) / float(rim_radius)

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
                self.release_angle_deg,
                self.samples,
            )
        return ideal_trajectory_pixels(
            release_xy,
            target_xy,
            self.release_angle_deg,
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
            horizontal_m = horizontal_radii * rim_radius_m
            vertical_m = vertical_radii * rim_radius_m
            velocity_scale_m = rim_radius_m
        straight_m = math.hypot(horizontal_m, vertical_m)
        target_speed = required_release_speed(
            horizontal_m,
            vertical_m,
            self.release_angle_deg,
        )

        observed_angle, observed_speed = self._observed_release_kinematics(
            velocity_scale_m
        )
        angle_error = (
            observed_angle - self.release_angle_deg
            if observed_angle is not None
            else None
        )
        speed_error = (
            observed_speed - target_speed
            if observed_speed is not None and target_speed is not None
            else None
        )

        path_rmse, observed_apex, ideal_apex = self._path_comparison()
        apex_error = (
            observed_apex - ideal_apex
            if observed_apex is not None and ideal_apex is not None
            else None
        )

        return TrajectoryComparison(
            target_release_angle_deg=self.release_angle_deg,
            observed_release_angle_deg=observed_angle,
            release_angle_error_deg=angle_error,
            target_release_speed_m_s=target_speed,
            observed_release_speed_m_s=observed_speed,
            release_speed_error_m_s=speed_error,
            release_height_m=self._release_height_m if calibrated else None,
            rim_height_m=self.rim_height_m if calibrated else None,
            horizontal_distance_m=horizontal_m,
            vertical_distance_m=vertical_m,
            straight_line_distance_m=straight_m,
            meters_per_pixel_at_release=(
                self._meters_per_pixel_at_release if calibrated else None
            ),
            velocity_calibration=(
                "player_height" if calibrated else "rim_diameter_fallback"
            ),
            release_anchor_source=self._release_anchor_source,
            path_rmse_rim_radii=path_rmse,
            observed_apex_height_rim_radii=observed_apex,
            ideal_apex_height_rim_radii=ideal_apex,
            apex_height_error_rim_radii=apex_error,
            rim_crossing_error_rim_radii=(
                self._crossing_relative_x - target_x
                if self._crossing_relative_x is not None
                else None
            ),
            observed_point_count=len(self._observed),
        )

    def _observed_release_kinematics(
        self, rim_radius_m: float
    ) -> Tuple[Optional[float], Optional[float]]:
        points = self._observed[: self.velocity_fit_points]
        if len(points) < 2:
            return None, None
        times = np.asarray(
            [(point.timestamp_ms - points[0].timestamp_ms) / 1000.0 for point in points],
            dtype=np.float64,
        )
        if times[-1] <= 0.0 or len(np.unique(times)) < 2:
            return None, None

        vx_radii_s = float(np.polyfit(times, [p.x for p in points], 1)[0])
        vy_radii_s = float(np.polyfit(times, [p.y for p in points], 1)[0])
        target_x = self._target_relative_xy[0]
        direction_to_rim = 1.0 if target_x >= self._release.x else -1.0
        velocity_toward_rim = direction_to_rim * vx_radii_s
        observed_angle = None
        if velocity_toward_rim > 1e-6:
            observed_angle = math.degrees(
                math.atan2(-vy_radii_s, velocity_toward_rim)
            )
        observed_speed = math.hypot(vx_radii_s, vy_radii_s) * rim_radius_m
        return observed_angle, observed_speed

    def _path_comparison(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        release = self._release
        target_x, target_y = self._target_relative_xy
        if release is None or abs(target_x - release.x) <= 1e-6:
            return None, None, None

        direction = 1.0 if target_x >= release.x else -1.0
        distance = abs(target_x - release.x)
        height_to_target = release.y - target_y
        theta = math.radians(self.release_angle_deg)
        physical = self._physical_distances()
        curvature = (
            distance * math.tan(theta) - height_to_target
        ) / (distance * distance)

        residuals: List[float] = []
        observed_heights: List[float] = []
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
            observed_heights.append(target_y - point.y)

        sample_x = np.linspace(0.0, distance, self.samples)
        ideal_y = np.asarray(
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
                for progress in sample_x
            ],
            dtype=np.float64,
        )
        ideal_apex = float(target_y - np.min(ideal_y))

        if len(residuals) < self.comparison_min_points:
            return None, None, ideal_apex
        rmse = math.sqrt(float(np.mean(np.square(residuals))))
        observed_apex = max(observed_heights)
        return rmse, observed_apex, ideal_apex

    def _physical_distances(self) -> Optional[Tuple[float, float]]:
        if (
            self._release is None
            or self._release_height_m is None
            or self._meters_per_pixel_at_release is None
            or self._release_rim_radius_px is None
        ):
            return None
        horizontal_pixels = (
            abs(self._target_relative_xy[0] - self._release.x)
            * self._release_rim_radius_px
        )
        horizontal_m = horizontal_pixels * self._meters_per_pixel_at_release
        vertical_m = self.rim_height_m - self._release_height_m
        if (
            horizontal_m <= 1e-6
            or vertical_m < self.minimum_vertical_difference_m
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
            self.release_angle_deg,
        )
        if speed is None or vertical_m <= 1e-6:
            return release.y
        x_m = progress_radii / distance_radii * horizontal_m
        height_m = (
            x_m * math.tan(theta)
            - 9.81 * x_m * x_m
            / (2.0 * speed * speed * math.cos(theta) ** 2)
        )
        return release.y - height_m * (release.y - target_y) / vertical_m

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
        )

    @staticmethod
    def _to_screen(
        relative_xy: Point, rim_center_xy: Point, rim_radius: float
    ) -> Point:
        return (
            float(rim_center_xy[0]) + float(relative_xy[0]) * rim_radius,
            float(rim_center_xy[1]) + float(relative_xy[1]) * rim_radius,
        )
