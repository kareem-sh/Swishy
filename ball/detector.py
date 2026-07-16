"""Per-frame basketball detection (Phase 6a)."""

from typing import List, Optional
import cv2
import numpy as np
import yaml

from ball.models import BallDetection


class BallDetector:
    """Locate the ball in each video frame."""

    def __init__(self, config_path: str = "config/ball.yaml"):
        """Initialize ball detector with configuration."""
        self.config = self._load_config(config_path)
        self.detection_method = self.config.get("method", "color_segmentation")
        
        # Color segmentation parameters
        self.hsv_lower = np.array(self.config.get("hsv_lower", [0, 50, 50]))
        self.hsv_upper = np.array(self.config.get("hsv_upper", [20, 255, 255]))
        self.min_radius = self.config.get("min_radius", 5)
        self.max_radius = self.config.get("max_radius", 50)
        self.min_circularity = self.config.get("min_circularity", 0.7)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.6)
        
        # YOLO backend (optional)
        self.yolo_model = None
        if self.detection_method == "yolo":
            self._load_yolo_model()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    def _load_yolo_model(self):
        """Load YOLO model for ball detection."""
        try:
            from ultralytics import YOLO
            model_path = self.config.get("yolo_model_path", "models/yolo_ball.pt")
            self.yolo_model = YOLO(model_path)
        except ImportError:
            print("Warning: YOLO not available. Falling back to color segmentation.")
            self.detection_method = "color_segmentation"
        except Exception as e:
            print(f"Warning: Failed to load YOLO model: {e}. Falling back to color segmentation.")
            self.detection_method = "color_segmentation"

    def detect(self, bgr_image: np.ndarray, frame_index: int, timestamp_ms: int) -> Optional[BallDetection]:
        """Return the best ball detection for this frame, or None."""
        if bgr_image is None or bgr_image.size == 0:
            return None

        if self.detection_method == "yolo" and self.yolo_model is not None:
            return self._detect_yolo(bgr_image, frame_index, timestamp_ms)
        else:
            return self._detect_color(bgr_image, frame_index, timestamp_ms)

    def _detect_color(self, bgr_image: np.ndarray, frame_index: int, timestamp_ms: int) -> Optional[BallDetection]:
        """Detect ball using HSV color segmentation."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_detection = None
        best_confidence = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * (self.min_radius ** 2):
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue

            confidence = min(1.0, circularity * (radius / self.max_radius))
            if confidence > best_confidence and confidence >= self.confidence_threshold:
                best_confidence = confidence
                best_detection = BallDetection(
                    x=int(x),
                    y=int(y),
                    radius=int(radius),
                    confidence=confidence,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms
                )

        return best_detection

    def _detect_yolo(self, bgr_image: np.ndarray, frame_index: int, timestamp_ms: int) -> Optional[BallDetection]:
        """Detect ball using YOLO model."""
        results = self.yolo_model(bgr_image, conf=self.confidence_threshold)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # Get the highest confidence detection
        best_box = results[0].boxes[0]
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(best_box.conf[0])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = max((x2 - x1), (y2 - y1)) // 2

        return BallDetection(
            x=center_x,
            y=center_y,
            radius=radius,
            confidence=confidence,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms
        )

    def detect_batch(self, frames: List[np.ndarray]) -> List[Optional[BallDetection]]:
        """Detect ball across multiple frames (offline / video mode)."""
        detections = []
        for i, frame in enumerate(frames):
            timestamp_ms = i * 33  # Assuming 30fps
            detection = self.detect(frame, i, timestamp_ms)
            detections.append(detection)
        return detections