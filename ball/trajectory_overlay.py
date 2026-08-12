"""Shot-scoped observed ball trajectory for video overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ball.models import BallSnapshot

Point = Tuple[float, float]


@dataclass(frozen=True)
class _RelativeTrajectoryPoint:
    x: float
    y: float
    timestamp_ms: int


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
        self._segments: List[List[_RelativeTrajectoryPoint]] = []
        self._active = False
        self._gap_before_next_point = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._segments.clear()
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
    ) -> None:
        if released_this_frame:
            self.reset()
            self._active = True

        if not self._active:
            return

        valid_rim = (
            rim_center_xy is not None
            and rim_radius is not None
            and math.isfinite(float(rim_radius))
            and float(rim_radius) > 1e-6
        )
        observed_snapshot = snapshot is not None and not snapshot.is_interpolated
        if not observed_snapshot or not valid_rim:
            self._gap_before_next_point = True
        else:
            rim_x, rim_y = rim_center_xy
            radius = float(rim_radius)
            relative_point = _RelativeTrajectoryPoint(
                x=(snapshot.x - rim_x) / radius,
                y=(snapshot.y - rim_y) / radius,
                timestamp_ms=snapshot.timestamp_ms,
            )
            self._append(relative_point)

        if shot_finished:
            self._active = False

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

    def _trim_oldest_points(self) -> None:
        excess = sum(len(segment) for segment in self._segments) - self.max_points
        while excess > 0 and self._segments:
            first = self._segments[0]
            remove_count = min(excess, len(first))
            del first[:remove_count]
            excess -= remove_count
            if not first:
                self._segments.pop(0)
