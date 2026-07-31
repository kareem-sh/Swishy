"""OpenCV NanoTrack wrapper for per-frame basketball tracking.

YOLO is still responsible for finding the ball. NanoTrack only follows a
known YOLO bounding box between the more expensive YOLO correction frames.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from ball.models import BallDetection


class NanoBallTracker:
    """Track one YOLO-acquired basketball with OpenCV NanoTrack."""

    def __init__(
        self,
        backbone_path: str | Path,
        neckhead_path: str | Path,
        *,
        minimum_box_size_px: float = 12.0,
        maximum_center_jump_px: float = 180.0,
        initial_box_scale: float = 1.35,
        minimum_size_ratio: float = 0.35,
        maximum_size_ratio: float = 2.85,
        minimum_aspect_ratio: float = 0.35,
        maximum_aspect_ratio: float = 2.85,
        device: str = "auto",
        cuda_fp16: bool = False,
    ) -> None:
        self.backbone_path = Path(backbone_path)
        self.neckhead_path = Path(neckhead_path)
        self.minimum_box_size_px = max(float(minimum_box_size_px), 2.0)
        self.maximum_center_jump_px = max(float(maximum_center_jump_px), 1.0)
        self.initial_box_scale = max(float(initial_box_scale), 1.0)
        self.minimum_size_ratio = max(float(minimum_size_ratio), 0.01)
        self.maximum_size_ratio = max(
            float(maximum_size_ratio), self.minimum_size_ratio
        )
        self.minimum_aspect_ratio = max(float(minimum_aspect_ratio), 0.01)
        self.maximum_aspect_ratio = max(
            float(maximum_aspect_ratio), self.minimum_aspect_ratio
        )
        requested_device = str(device).strip().lower()
        if requested_device not in {"auto", "cpu", "gpu", "cuda"}:
            print(
                f"Warning: unknown NanoTrack device {requested_device!r}; "
                "using auto"
            )
            requested_device = "auto"
        self.device = "cpu"
        if requested_device in {"auto", "gpu", "cuda"}:
            if self._cuda_device_count() > 0:
                self.device = "cuda"
            elif requested_device in {"gpu", "cuda"}:
                print(
                    "NanoTrack CUDA requested but this OpenCV build has no "
                    "CUDA device; falling back to CPU"
                )
        self.cuda_fp16 = bool(cuda_fp16)

        self._tracker = None
        self._active = False
        self._previous_center: Optional[Tuple[float, float]] = None
        self._previous_size: Optional[Tuple[float, float]] = None
        self._last_score = 0.0

        if not hasattr(cv2, "TrackerNano_create"):
            raise RuntimeError(
                "OpenCV NanoTrack is unavailable. Install opencv-contrib-python."
            )
        if not self.backbone_path.is_file():
            raise FileNotFoundError(
                f"NanoTrack backbone model not found: {self.backbone_path}"
            )
        if not self.neckhead_path.is_file():
            raise FileNotFoundError(
                f"NanoTrack neck/head model not found: {self.neckhead_path}"
            )
        self._tracker = self._create_tracker()
        print(f"NanoTrack device: {self.device.upper()}")

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_score(self) -> float:
        return self._last_score

    def reset(self) -> None:
        # Keep the loaded networks in memory. The same OpenCV tracker can be
        # initialized again when YOLO reacquires or corrects the ball.
        self._active = False
        self._previous_center = None
        self._previous_size = None
        self._last_score = 0.0

    def initialize(
        self,
        frame: np.ndarray,
        detection: BallDetection,
    ) -> bool:
        """Start or correct NanoTrack from a YOLO ball bounding box."""
        if frame is None or frame.size == 0:
            self.reset()
            return False

        init_box = self._expanded_xywh(detection.bbox_xyxy, frame.shape)
        if init_box is None:
            self.reset()
            return False

        try:
            self._tracker.init(frame, init_box)
        except cv2.error:
            try:
                if self.device == "cuda":
                    print(
                        "Warning: NanoTrack CUDA initialization failed; "
                        "falling back to CPU"
                    )
                    self.device = "cpu"
                self._tracker = self._create_tracker()
                self._tracker.init(frame, init_box)
            except cv2.error:
                self.reset()
                return False

        x, y, width, height = init_box
        self._active = True
        self._previous_center = (x + width / 2.0, y + height / 2.0)
        self._previous_size = (float(width), float(height))
        self._last_score = float(detection.confidence)
        return True

    def update(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallDetection]:
        """Follow the initialized ball for one frame."""
        if not self._active or frame is None or frame.size == 0:
            return None

        try:
            found, raw_box = self._tracker.update(frame)
        except cv2.error:
            if self.device == "cuda":
                print(
                    "Warning: NanoTrack CUDA update failed; falling back to CPU"
                )
                self.device = "cpu"
                try:
                    self._tracker = self._create_tracker()
                except cv2.error:
                    pass
            self.reset()
            return None

        if not found:
            self.reset()
            return None

        x, y, width, height = map(float, raw_box)
        if not self._is_valid_box(x, y, width, height, frame.shape):
            self.reset()
            return None

        center = (x + width / 2.0, y + height / 2.0)
        if self._previous_center is not None:
            center_jump = math.dist(center, self._previous_center)
            if center_jump > self.maximum_center_jump_px:
                self.reset()
                return None

        if self._previous_size is not None:
            previous_area = self._previous_size[0] * self._previous_size[1]
            current_area = width * height
            size_ratio = current_area / max(previous_area, 1.0)
            if not self.minimum_size_ratio <= size_ratio <= self.maximum_size_ratio:
                self.reset()
                return None

        self._previous_center = center
        self._previous_size = (width, height)
        self._last_score = self._tracking_score()

        return BallDetection(
            center_xy=center,
            bbox_xyxy=(x, y, x + width, y + height),
            confidence=self._last_score,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _tracking_score(self) -> float:
        try:
            score = float(self._tracker.getTrackingScore())
        except (AttributeError, cv2.error, TypeError, ValueError):
            score = 0.25
        if not math.isfinite(score):
            return 0.25
        return min(max(score, 0.0), 1.0)

    def _create_tracker(self):
        params = cv2.TrackerNano_Params()
        params.backbone = str(self.backbone_path)
        params.neckhead = str(self.neckhead_path)
        if self.device == "cuda":
            params.backend = cv2.dnn.DNN_BACKEND_CUDA
            params.target = (
                cv2.dnn.DNN_TARGET_CUDA_FP16
                if self.cuda_fp16
                else cv2.dnn.DNN_TARGET_CUDA
            )
        else:
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
        return cv2.TrackerNano_create(params)

    @staticmethod
    def _cuda_device_count() -> int:
        try:
            return (
                int(cv2.cuda.getCudaEnabledDeviceCount())
                if hasattr(cv2, "cuda")
                else 0
            )
        except cv2.error:
            return 0

    def _expanded_xywh(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        frame_shape,
    ) -> Optional[Tuple[int, int, int, int]]:
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = map(float, bbox_xyxy)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = max(x2 - x1, self.minimum_box_size_px) * self.initial_box_scale
        height = max(y2 - y1, self.minimum_box_size_px) * self.initial_box_scale

        x1 = max(0.0, center_x - width / 2.0)
        y1 = max(0.0, center_y - height / 2.0)
        x2 = min(float(frame_width), center_x + width / 2.0)
        y2 = min(float(frame_height), center_y + height / 2.0)
        width = x2 - x1
        height = y2 - y1

        if width < self.minimum_box_size_px or height < self.minimum_box_size_px:
            return None
        return (
            int(round(x1)),
            int(round(y1)),
            int(round(width)),
            int(round(height)),
        )

    def _is_valid_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        frame_shape,
    ) -> bool:
        frame_height, frame_width = frame_shape[:2]
        values = (x, y, width, height)
        if not all(math.isfinite(value) for value in values):
            return False
        if width < self.minimum_box_size_px or height < self.minimum_box_size_px:
            return False
        if x + width <= 0 or y + height <= 0 or x >= frame_width or y >= frame_height:
            return False

        aspect_ratio = width / max(height, 1e-6)
        return self.minimum_aspect_ratio <= aspect_ratio <= self.maximum_aspect_ratio
