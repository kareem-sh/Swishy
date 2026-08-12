"""Make / miss / unknown classification (Phase 6c)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ball.models import BallTrajectory, ShotOutcome
from ball.timeseries import BallTimeSeriesBuffer
from utils.config_loader import load_yaml


class OutcomeClassifier:
    """Rule-based shot outcome from ball time-series and hoop ROI."""

    def __init__(self, config_name: str = "ball.yaml"):
        self.config = load_yaml(config_name)
        outcome = self.config.get("outcome", {})
        self.hoop_roi = None
        self.rim_radius_px = float(
            outcome.get("rim_radius_px", self.config.get("rim_radius_px", 30))
        )
        self.max_rim_offset = float(
            outcome.get("max_rim_offset", self.config.get("max_rim_offset", 20))
        )
        self.rim_inner_scale = float(outcome.get("rim_inner_scale", 0.72))
        self.confidence_threshold = float(
            outcome.get(
                "confidence_threshold",
                self.config.get("outcome_confidence_threshold", 0.6),
            )
        )
        self.load_hoop_roi("hoop_roi.yaml")

    def load_hoop_roi(self, config_name: str = "hoop_roi.yaml") -> bool:
        """Load hoop region from YAML (normalized or pixel)."""
        data = load_yaml(config_name)
        if not data:
            self.hoop_roi = None
            return False

        hoop = data.get("hoop", data)
        if not hoop or not hoop.get("enabled", False):
            self.hoop_roi = None
            return False

        roi = hoop.get("roi_normalized")
        if roi and len(roi) == 4:
            x1, y1, x2, y2 = roi
            self.hoop_roi = {
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
                "rim_radius": max(x2 - x1, y2 - y1) / 2.0,
                "roi_normalized": roi,
                "rim_inner_scale": float(hoop.get("rim_inner_scale", 0.72)),
                "source": "config_normalized",
            }
            return True

        self.hoop_roi = hoop
        return True

    def set_rim_from_detection(
        self,
        center_xy: Tuple[float, float],
        bbox_xyxy: Tuple[float, float, float, float],
    ) -> None:
        """Update hoop ROI from a live rim detection (pixels)."""
        x1, y1, x2, y2 = bbox_xyxy
        # Hoop detections often include the hanging net, so vertical box size
        # is not a useful rim radius. Use the horizontal opening instead.
        rim_radius = (x2 - x1) * self.rim_inner_scale / 2.0
        self.hoop_roi = {
            "center_x": float(center_xy[0]),
            "center_y": float(center_xy[1]),
            "rim_radius": float(rim_radius),
            "bbox_xyxy": bbox_xyxy,
            "source": "yolo",
        }

    def classify(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        trajectory: Optional[BallTrajectory] = None,
    ) -> ShotOutcome:
        if self.hoop_roi is None:
            return ShotOutcome(
                result="unknown",
                confidence=0.0,
                evidence=["Hoop ROI not configured"],
            )

        if len(ball_buffer) < 5:
            return ShotOutcome(
                result="unknown",
                confidence=0.0,
                evidence=["Insufficient ball data"],
            )

        rim_snapshots = self._get_rim_snapshots(ball_buffer)
        if not rim_snapshots:
            return ShotOutcome(
                result="unknown",
                confidence=0.0,
                evidence=["Ball did not approach rim"],
            )

        result, confidence, evidence = self._analyze_rim_approach(rim_snapshots)
        if trajectory is not None:
            result, confidence, evidence = self._refine_with_trajectory(
                result, confidence, evidence, trajectory
            )

        return ShotOutcome(result=result, confidence=confidence, evidence=evidence)

    def _get_rim_snapshots(self, buffer: BallTimeSeriesBuffer) -> list:
        if not buffer.buffer or self.hoop_roi is None:
            return []

        hoop_center_x = float(self.hoop_roi.get("center_x", 0))
        hoop_center_y = float(self.hoop_roi.get("center_y", 0))
        rim_radius = float(self.hoop_roi.get("rim_radius", self.rim_radius_px))

        rim_snapshots = []
        for snapshot in buffer.buffer:
            dist = np.hypot(snapshot.x - hoop_center_x, snapshot.y - hoop_center_y)
            if dist <= rim_radius * 1.5:
                rim_snapshots.append((snapshot, dist))
        return rim_snapshots

    def _analyze_rim_approach(
        self, rim_snapshots: list
    ) -> Tuple[str, float, List[str]]:
        if not rim_snapshots:
            return "unknown", 0.0, ["No snapshots near rim"]

        rim_radius = float(self.hoop_roi.get("rim_radius", self.rim_radius_px))
        inside_rim = any(dist < rim_radius for _, dist in rim_snapshots)
        close_to_rim = any(dist < rim_radius * 1.2 for _, dist in rim_snapshots)

        if len(rim_snapshots) >= 2:
            y_vel = rim_snapshots[-1][0].y - rim_snapshots[0][0].y
            if y_vel > 0:
                if inside_rim:
                    return "made", 0.85, ["Ball entered rim area"]
                if close_to_rim:
                    _, dist = rim_snapshots[-1]
                    offset = dist - rim_radius
                    if offset <= self.max_rim_offset:
                        return "missed", 0.65, ["Ball close to rim but outside"]
                    return "missed", 0.75, ["Ball significantly outside rim"]

        return "unknown", 0.3, ["Insufficient evidence"]

    def _refine_with_trajectory(
        self,
        result: str,
        confidence: float,
        evidence: List[str],
        trajectory: BallTrajectory,
    ) -> Tuple[str, float, List[str]]:
        if trajectory is None or trajectory.fit_params is None:
            return result, confidence, evidence

        entry_angle = trajectory.entry_angle_deg
        if entry_angle is None:
            return result, confidence, evidence

        if 40 <= entry_angle <= 55 and result == "made":
            confidence = min(1.0, confidence + 0.1)
            evidence = evidence + ["good entry angle"]
        elif result == "missed" and (entry_angle < 30 or entry_angle > 60):
            confidence = min(1.0, confidence + 0.1)
            evidence = evidence + ["poor entry angle"]

        return result, confidence, evidence
