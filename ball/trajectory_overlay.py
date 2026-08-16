"""Shot-scoped observed ball trajectory for video overlays."""

from __future__ import annotations

import math
<<<<<<< HEAD
from dataclasses import dataclass
from typing import List, Optional, Tuple
=======
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence, Tuple
>>>>>>> balltraking-fixedpose

import numpy as np

from ball.models import BallSnapshot

Point = Tuple[float, float]


@dataclass(frozen=True)
class _RelativeTrajectoryPoint:
    x: float
    y: float
    timestamp_ms: int
<<<<<<< HEAD
=======
    wrist_distance_px: Optional[float] = None
    wrist_relative_xy: Optional[Point] = None
    measurement_source: str = "unknown"
    is_direct_observation: bool = True
>>>>>>> balltraking-fixedpose


class ObservedTrajectoryRecorder:
    """Store measured ball points relative to the moving rim.

    Rim-relative coordinates keep old points aligned when a handheld camera
    translates or zooms. Missing observations split the trail so the renderer
    never draws a fake connecting line across a detection gap.
    """

    def __init__(
        self,
        *,
        max_points: int = 120,
        maximum_jump_rim_radii: float = 5.0,
        fit_min_points: int = 5,
        fit_samples: int = 60,
        fit_outlier_threshold_rim_radii: float = 0.75,
<<<<<<< HEAD
=======
        release_preroll_s: float = 0.4,
        release_near_wrist_scale: float = 0.5,
>>>>>>> balltraking-fixedpose
    ) -> None:
        self.max_points = max(2, int(max_points))
        self.maximum_jump_rim_radii = max(
            0.1, float(maximum_jump_rim_radii)
        )
        self.fit_min_points = max(3, int(fit_min_points))
        self.fit_samples = max(10, int(fit_samples))
        self.fit_outlier_threshold_rim_radii = max(
            0.05, float(fit_outlier_threshold_rim_radii)
        )
<<<<<<< HEAD
        self._segments: List[List[_RelativeTrajectoryPoint]] = []
=======
        self.release_near_wrist_scale = max(
            0.05, float(release_near_wrist_scale)
        )
        self.release_preroll_ms = max(0, int(float(release_preroll_s) * 1000))
        self._segments: List[List[_RelativeTrajectoryPoint]] = []
        self._preroll: Deque[_RelativeTrajectoryPoint] = deque()
        self._kinematics_start_timestamp_ms: Optional[int] = None
>>>>>>> balltraking-fixedpose
        self._active = False
        self._gap_before_next_point = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._segments.clear()
<<<<<<< HEAD
=======
        self._preroll.clear()
        self._kinematics_start_timestamp_ms = None
>>>>>>> balltraking-fixedpose
        self._active = False
        self._gap_before_next_point = False

    def update(
        self,
        snapshot: Optional[BallSnapshot],
        *,
        released_this_frame: bool,
        shot_finished: bool,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
<<<<<<< HEAD
    ) -> None:
        if released_this_frame:
            self.reset()
            self._active = True

        if not self._active:
            return

=======
        wrist_xy: Optional[Point] = None,
        release_distance_px: float = 60.0,
    ) -> None:
>>>>>>> balltraking-fixedpose
        valid_rim = (
            rim_center_xy is not None
            and rim_radius is not None
            and math.isfinite(float(rim_radius))
            and float(rim_radius) > 1e-6
        )
<<<<<<< HEAD
        observed_snapshot = snapshot is not None and not snapshot.is_interpolated
        if not observed_snapshot or not valid_rim:
            self._gap_before_next_point = True
        else:
            rim_x, rim_y = rim_center_xy
            radius = float(rim_radius)
=======
        observed_snapshot = (
            snapshot is not None and snapshot.is_visual_observation
        )
        relative_point = None
        if observed_snapshot and valid_rim:
            rim_x, rim_y = rim_center_xy
            radius = float(rim_radius)
            wrist_distance = None
            wrist_relative = None
            if wrist_xy is not None:
                wrist_distance = math.hypot(
                    snapshot.x - float(wrist_xy[0]),
                    snapshot.y - float(wrist_xy[1]),
                )
                wrist_relative = (
                    (float(wrist_xy[0]) - rim_x) / radius,
                    (float(wrist_xy[1]) - rim_y) / radius,
                )
>>>>>>> balltraking-fixedpose
            relative_point = _RelativeTrajectoryPoint(
                x=(snapshot.x - rim_x) / radius,
                y=(snapshot.y - rim_y) / radius,
                timestamp_ms=snapshot.timestamp_ms,
<<<<<<< HEAD
            )
=======
                wrist_distance_px=wrist_distance,
                wrist_relative_xy=wrist_relative,
                measurement_source=snapshot.measurement_source,
                is_direct_observation=snapshot.is_direct_observation,
            )
            if (
                not self._preroll
                or relative_point.timestamp_ms != self._preroll[-1].timestamp_ms
            ):
                self._preroll.append(relative_point)
                self._trim_preroll(relative_point.timestamp_ms)

        if released_this_frame:
            self._segments.clear()
            self._gap_before_next_point = False
            selected = self._release_preroll(float(release_distance_px))
            if selected:
                self._segments.append(list(selected))
                self._kinematics_start_timestamp_ms = (
                    self._release_kinematics_start(
                        selected,
                        float(release_distance_px),
                    )
                )
            self._active = True
        elif not self._active:
            return
        elif relative_point is None:
            self._gap_before_next_point = True
        else:
>>>>>>> balltraking-fixedpose
            self._append(relative_point)

        if shot_finished:
            self._active = False

<<<<<<< HEAD
=======
    def relative_points(self) -> List[Tuple[float, float, int]]:
        """Return recorded points for calculations in rim-radius units."""
        return [
            (point.x, point.y, point.timestamp_ms)
            for segment in self._segments
            for point in segment
        ]

    def relative_measurements(
        self,
    ) -> List[Tuple[float, float, int, str, bool]]:
        """Return points plus detector provenance for numeric calculations."""
        return [
            (
                point.x,
                point.y,
                point.timestamp_ms,
                point.measurement_source,
                point.is_direct_observation,
            )
            for segment in self._segments
            for point in segment
        ]

    def release_wrist_context(self) -> Tuple[Optional[Point], Optional[float]]:
        """Return the wrist recorded with the reconstructed release point."""
        if not self._segments or not self._segments[0]:
            return None, None
        release = self._segments[0][0]
        return release.wrist_relative_xy, release.wrist_distance_px

    def release_kinematics_start_timestamp_ms(self) -> Optional[int]:
        """First observed frame after the ball has separated from the hand."""
        return self._kinematics_start_timestamp_ms

>>>>>>> balltraking-fixedpose
    def screen_segments(
        self,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
    ) -> List[List[Point]]:
        if (
            rim_center_xy is None
            or rim_radius is None
            or not math.isfinite(float(rim_radius))
            or float(rim_radius) <= 1e-6
        ):
            return []

        rim_x, rim_y = rim_center_xy
        radius = float(rim_radius)
        return [
            [
                (rim_x + point.x * radius, rim_y + point.y * radius)
                for point in segment
            ]
            for segment in self._segments
            if segment
        ]

    def fitted_screen_curve(
        self,
        rim_center_xy: Optional[Point],
        rim_radius: Optional[float],
    ) -> List[Point]:
        """Fit x(t) linearly and y(t) quadratically with robust rejection."""
        if (
            rim_center_xy is None
            or rim_radius is None
            or not math.isfinite(float(rim_radius))
            or float(rim_radius) <= 1e-6
        ):
            return []

        points = [point for segment in self._segments for point in segment]
        if len(points) < self.fit_min_points:
            return []

        timestamps = np.asarray(
            [point.timestamp_ms for point in points], dtype=np.float64
        )
        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            return []
        time_normalized = (timestamps - timestamps[0]) / duration
        x_values = np.asarray([point.x for point in points], dtype=np.float64)
        y_values = np.asarray([point.y for point in points], dtype=np.float64)
        inliers = np.ones(len(points), dtype=bool)

        for _ in range(3):
            if int(np.count_nonzero(inliers)) < self.fit_min_points:
                return []
            x_coefficients = np.polyfit(
                time_normalized[inliers], x_values[inliers], 1
            )
            y_coefficients = np.polyfit(
                time_normalized[inliers], y_values[inliers], 2
            )
            predicted_x = np.polyval(x_coefficients, time_normalized)
            predicted_y = np.polyval(y_coefficients, time_normalized)
            residuals = np.hypot(
                x_values - predicted_x,
                y_values - predicted_y,
            )
            median = float(np.median(residuals[inliers]))
            mad = float(np.median(np.abs(residuals[inliers] - median)))
            robust_sigma = max(1.4826 * mad, 0.02)
            threshold = max(
                self.fit_outlier_threshold_rim_radii,
                median + 3.5 * robust_sigma,
            )
            updated_inliers = residuals <= threshold
            if np.array_equal(updated_inliers, inliers):
                break
            inliers = updated_inliers

        if int(np.count_nonzero(inliers)) < self.fit_min_points:
            return []
        x_coefficients = np.polyfit(
            time_normalized[inliers], x_values[inliers], 1
        )
        y_coefficients = np.polyfit(
            time_normalized[inliers], y_values[inliers], 2
        )
        sample_times = np.linspace(0.0, 1.0, self.fit_samples)
        fitted_x = np.polyval(x_coefficients, sample_times)
        fitted_y = np.polyval(y_coefficients, sample_times)

        rim_x, rim_y = rim_center_xy
        radius = float(rim_radius)
        return [
            (rim_x + float(x) * radius, rim_y + float(y) * radius)
            for x, y in zip(fitted_x, fitted_y)
        ]

    def _append(self, point: _RelativeTrajectoryPoint) -> None:
        start_new_segment = self._gap_before_next_point or not self._segments
        if not start_new_segment and self._segments[-1]:
            previous = self._segments[-1][-1]
            start_new_segment = (
                math.hypot(previous.x - point.x, previous.y - point.y)
                > self.maximum_jump_rim_radii
            )

        if start_new_segment:
            self._segments.append([])
        self._segments[-1].append(point)
        self._gap_before_next_point = False
        self._trim_oldest_points()

<<<<<<< HEAD
=======
    def _release_preroll(
        self, release_distance_px: float
    ) -> Sequence[_RelativeTrajectoryPoint]:
        """Backtrack confirmation to the last near-hand ball position."""
        points = list(self._preroll)
        if not points:
            return []

        near_limit = max(1.0, release_distance_px * self.release_near_wrist_scale)
        selected_index = len(points) - 1
        for index in range(len(points) - 2, -1, -1):
            distance = points[index].wrist_distance_px
            later = [
                point.wrist_distance_px
                for point in points[index:]
                if point.wrist_distance_px is not None
            ]
            if (
                distance is not None
                and distance <= near_limit
                and len(later) >= 2
                and later[-1] > later[0]
            ):
                selected_index = index
                break
        return points[selected_index:]

    @staticmethod
    def _release_kinematics_start(
        points: Sequence[_RelativeTrajectoryPoint],
        release_distance_px: float,
    ) -> int:
        """Choose the first post-separation point, excluding drawing preroll."""
        separation_limit = max(1.0, release_distance_px)
        for point in points:
            if (
                point.wrist_distance_px is not None
                and point.wrist_distance_px >= separation_limit
            ):
                return point.timestamp_ms

        # Missing/low-confidence wrist landmarks must not make the pre-release
        # hand motion look like launch velocity. The confirmation frame is the
        # safest fallback because the ball FSM has already accepted release.
        return points[-1].timestamp_ms

>>>>>>> balltraking-fixedpose
    def _trim_oldest_points(self) -> None:
        excess = sum(len(segment) for segment in self._segments) - self.max_points
        while excess > 0 and self._segments:
            first = self._segments[0]
            remove_count = min(excess, len(first))
            del first[:remove_count]
            excess -= remove_count
            if not first:
                self._segments.pop(0)
<<<<<<< HEAD
=======

    def _trim_preroll(self, newest_timestamp_ms: int) -> None:
        while (
            len(self._preroll) > 1
            and newest_timestamp_ms - self._preroll[0].timestamp_ms
            > self.release_preroll_ms
        ):
            self._preroll.popleft()
>>>>>>> balltraking-fixedpose
