"""Per-frame basketball detection (Phase 6a)."""

from typing import List, Optional

import numpy as np

from ball.models import BallDetection


class BallDetector:
    """Locate the ball in each video frame."""

    def __init__(self):
        # TODO: load detector backend from config/ball.yaml (color segmentation or YOLO)
        pass

    def detect(self, bgr_image: np.ndarray, frame_index: int, timestamp_ms: int) -> Optional[BallDetection]:
        """Return the best ball detection for this frame, or None."""
        # TODO: run HSV color mask or YOLO inference
        # TODO: filter by size, circularity, and confidence threshold
        pass

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[BallDetection]]:
        """Detect ball across multiple frames (offline / video mode)."""
        # TODO: batch inference for throughput
        pass
