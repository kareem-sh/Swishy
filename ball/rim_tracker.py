"""Motion-aware helpers for a rim observed by a moving camera."""

from __future__ import annotations

import math
from typing import Optional

from ball.models import RimDetection


class RimMotionSmoother:
    """Smooth rim motion without assuming that it stays at fixed pixels."""

    def __init__(
        self,
        *,
        position_alpha: float = 0.75,
        size_alpha: float = 0.50,
        center_y_fraction: float = 0.50,
    ) -> None:
        self.position_alpha = min(max(float(position_alpha), 0.0), 1.0)
        self.size_alpha = min(max(float(size_alpha), 0.0), 1.0)
        self.center_y_fraction = min(max(float(center_y_fraction), 0.0), 1.0)
        self.current: Optional[RimDetection] = None

    def reset(self) -> None:
        self.current = None

    def update(
        self,
        detection: RimDetection,
        *,
        snap: bool = False,
    ) -> RimDetection:
        """Return a stabilized box, snapping immediately after reacquisition."""
        x1, y1, x2, y2 = detection.bbox_xyxy
        measured_width = max(float(x2 - x1), 1.0)
        measured_height = max(float(y2 - y1), 1.0)
        measured_x, measured_y = map(float, detection.center_xy)

        if self.current is None or snap:
            center_x = measured_x
            center_y = measured_y
            width = measured_width
            height = measured_height
        else:
            old_x1, old_y1, old_x2, old_y2 = self.current.bbox_xyxy
            old_width = max(float(old_x2 - old_x1), 1.0)
            old_height = max(float(old_y2 - old_y1), 1.0)
            old_x, old_y = self.current.center_xy
            pa = self.position_alpha
            sa = self.size_alpha
            center_x = pa * measured_x + (1.0 - pa) * old_x
            center_y = pa * measured_y + (1.0 - pa) * old_y
            width = sa * measured_width + (1.0 - sa) * old_width
            height = sa * measured_height + (1.0 - sa) * old_height

        left = center_x - width / 2.0
        top = center_y - height * self.center_y_fraction
        stabilized = RimDetection(
            center_xy=(center_x, center_y),
            bbox_xyxy=(left, top, left + width, top + height),
            confidence=float(detection.confidence),
            frame_index=detection.frame_index,
            timestamp_ms=detection.timestamp_ms,
        )
        self.current = stabilized
        return stabilized


def is_plausible_rim_correction(
    candidate: RimDetection,
    tracked: RimDetection,
    *,
    maximum_center_distance_px: float,
    maximum_center_distance_scale: float,
    maximum_width_change_ratio: float,
) -> bool:
    """Reject a YOLO correction that disagrees sharply with the live tracker."""
    cx1, _, cx2, _ = candidate.bbox_xyxy
    tx1, _, tx2, _ = tracked.bbox_xyxy
    candidate_width = max(float(cx2 - cx1), 1.0)
    tracked_width = max(float(tx2 - tx1), 1.0)
    ratio = candidate_width / tracked_width
    maximum_ratio = max(float(maximum_width_change_ratio), 1.0)
    if not 1.0 / maximum_ratio <= ratio <= maximum_ratio:
        return False

    allowed_distance = max(
        float(maximum_center_distance_px),
        tracked_width * float(maximum_center_distance_scale),
    )
    return math.dist(candidate.center_xy, tracked.center_xy) <= allowed_distance
