"""Per-frame basketball + rim detection (Phase 6a).

Uses the custom E-BARD YOLOv8n model (basketball + hoop), not COCO.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ball.models import BallDetection, CourtDetections, RimDetection
from ball.yolo_model import (
    DEFAULT_HF_FILE,
    DEFAULT_HF_REPO,
    DEFAULT_LOCAL,
    load_basketball_yolo,
    resolve_device,
)
from utils.config_loader import load_yaml


def _pick_best(boxes: Sequence[Tuple[Tuple[float, float, float, float], float]]):
    if not boxes:
        return None
    return max(boxes, key=lambda item: item[1])


class BallDetector:
    """Detect ball + rim each frame with the custom basketball YOLO."""

    def __init__(self, config_name: str = "ball.yaml"):
        raw = load_yaml(config_name)
        det = raw.get("detector", raw)

        self.enabled = bool(raw.get("enabled", True))
        self.backend = str(det.get("backend", "yolo")).lower()
        self.min_confidence = float(det.get("min_confidence", 0.12))
        self.min_confidence_rim = float(
            det.get("min_confidence_rim", min(self.min_confidence, 0.08))
        )
        self.imgsz = int(det.get("imgsz", 704))
        self.fallback_imgsz = int(det.get("fallback_imgsz", 512))
        self.fallback_on_ball_miss = bool(
            det.get("fallback_on_ball_miss", False)
        )
        self.fallback_until_rim_found = bool(
            det.get("fallback_until_rim_found", True)
        )
        self.rim_search_retry_interval = max(
            0,
            int(det.get("rim_search_retry_interval", 4)),
        )
        self.device = resolve_device(det.get("device", "auto"))
        self.frame_stride = max(1, int(det.get("frame_stride", 1)))
        self.min_radius = float(det.get("min_radius_px", 4))
        self.max_radius = float(det.get("max_radius_px", 140))
        self.sticky_rim = bool(det.get("sticky_rim", True))
        self.rim_center_y_fraction = float(
            det.get("rim_center_y_fraction", 0.15)
        )
        self.rim_lock_max_shift_ratio = float(
            det.get("rim_lock_max_shift_ratio", 0.75)
        )

        self.model_path = str(det.get("model_path") or DEFAULT_LOCAL.as_posix())
        self.hf_repo = str(det.get("hf_repo", DEFAULT_HF_REPO))
        self.hf_file = str(det.get("hf_file", DEFAULT_HF_FILE))
        self.use_quantize = (
            bool(det.get("quantize", 16))
            and self.model_path.lower().endswith(".pt")
            and self.device != "cpu"
        )
        self.rim_refresh_frames = max(
            1,
            int(det.get("rim_refresh_frames", 30)),
        )
        self._model = None
        self._ball_ids: List[int] = []
        self._rim_ids: List[int] = []
        self._frame_counter = 0
        self._last = CourtDetections()
        self._sticky_rim_det: Optional[RimDetection] = None

        if self.enabled and self.backend == "yolo":
            self._model, self._ball_ids, self._rim_ids, _ = load_basketball_yolo(
                self.model_path,
                hf_repo=self.hf_repo,
                hf_file=self.hf_file,
            )
            print(f"YOLO device: {self.device}")
        elif self.enabled and self.backend == "color":
            self.hsv_lower = np.array(det.get("hsv_lower", [5, 80, 80]), dtype=np.uint8)
            self.hsv_upper = np.array(det.get("hsv_upper", [25, 255, 255]), dtype=np.uint8)
            self.min_circularity = float(det.get("min_circularity", 0.65))

    @property
    def ready(self) -> bool:
        return self.enabled and (
            (self.backend == "yolo" and self._model is not None)
            or self.backend == "color"
        )

    def reset(self) -> None:
        self._frame_counter = 0
        self._last = CourtDetections()
        self._sticky_rim_det = None

    def detect(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallDetection]:
        return self.detect_court(bgr_image, frame_index, timestamp_ms).ball

    def detect_court(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> CourtDetections:
        if not self.ready or bgr_image is None or bgr_image.size == 0:
            return CourtDetections()

        self._frame_counter += 1
        if self.frame_stride > 1 and (self._frame_counter - 1) % self.frame_stride != 0:
            return CourtDetections(
                ball=self._retag(self._last.ball, frame_index, timestamp_ms),
                rim=self._retag_rim(
                    self._sticky_rim_det or self._last.rim, frame_index, timestamp_ms
                ),
            )

        if self.backend == "yolo":
            result = self._detect_yolo(bgr_image, frame_index, timestamp_ms)
        else:
            result = CourtDetections(
                ball=self._detect_color(bgr_image, frame_index, timestamp_ms)
            )

        if self.sticky_rim and result.rim is not None:
            if (
                self._sticky_rim_det is None
                or result.rim.confidence >= self._sticky_rim_det.confidence
            ):
                self._sticky_rim_det = result.rim
        elif self.sticky_rim and self._sticky_rim_det is not None:
            result = CourtDetections(
                ball=result.ball,
                rim=self._retag_rim(self._sticky_rim_det, frame_index, timestamp_ms),
            )

        self._last = result
        return result

    @staticmethod
    def _retag(
        det: Optional[BallDetection], frame_index: int, timestamp_ms: int
    ) -> Optional[BallDetection]:
        if det is None:
            return None
        return BallDetection(
            center_xy=det.center_xy,
            bbox_xyxy=det.bbox_xyxy,
            confidence=det.confidence * 0.9,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            measurement_source="reused",
        )

    @staticmethod
    def _retag_rim(
        det: Optional[RimDetection], frame_index: int, timestamp_ms: int
    ) -> Optional[RimDetection]:
        if det is None:
            return None
        return RimDetection(
            center_xy=det.center_xy,
            bbox_xyxy=det.bbox_xyxy,
            confidence=det.confidence * 0.95,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
        )

    def _detect_yolo(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> CourtDetections:
        searching_for_rim = (
            self.fallback_until_rim_found
            and self._sticky_rim_det is None
            and self.fallback_imgsz > 0
            and self.fallback_imgsz != self.imgsz
        )
        prediction_imgsz = self.imgsz
        if searching_for_rim:
            prediction_imgsz = self.fallback_imgsz

        result = self._predict_yolo(
            bgr_image,
            frame_index,
            timestamp_ms,
            prediction_imgsz,
        )
        result = self._reject_rim_jump(result)

        needs_ball = self.fallback_on_ball_miss and result.ball is None
        needs_rim_retry = (
            searching_for_rim
            and result.rim is None
            and self.rim_search_retry_interval > 0
            and self._frame_counter % self.rim_search_retry_interval == 0
        )
        if (
            self.fallback_imgsz > 0
            and self.fallback_imgsz != self.imgsz
            and (needs_ball or needs_rim_retry)
        ):
            retry_imgsz = (
                self.imgsz
                if needs_rim_retry
                else self.fallback_imgsz
            )
            fallback = self._predict_yolo(
                bgr_image,
                frame_index,
                timestamp_ms,
                retry_imgsz,
            )
            fallback = self._reject_rim_jump(fallback)
            result = CourtDetections(
                ball=result.ball or fallback.ball,
                rim=result.rim or fallback.rim,
            )

        return result

    def _reject_rim_jump(self, result: CourtDetections) -> CourtDetections:
        """Keep a locked static rim from jumping to a distant false detection."""
        if (
            not getattr(self, "sticky_rim", True)
            or self._sticky_rim_det is None
            or result.rim is None
        ):
            return result

        locked = self._sticky_rim_det
        lx1, ly1, lx2, ly2 = locked.bbox_xyxy
        locked_size = max(lx2 - lx1, ly2 - ly1)
        max_shift = max(40.0, locked_size * self.rim_lock_max_shift_ratio)
        center_shift = float(
            np.linalg.norm(
                np.asarray(result.rim.center_xy) - np.asarray(locked.center_xy)
            )
        )
        if center_shift > max_shift:
            return CourtDetections(ball=result.ball)
        return result

    def _predict_yolo(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
        imgsz: int,
    ) -> CourtDetections:
        """Run one YOLO pass at a specific input resolution."""
        detect_rim_now = (
            self._sticky_rim_det is None
            or (self._frame_counter - 1) % self.rim_refresh_frames == 0
        )
        requested_classes = list(self._ball_ids)

        if detect_rim_now:
            requested_classes.extend(self._rim_ids)

        predict_args = dict(
            source=bgr_image,
            conf=min(self.min_confidence, self.min_confidence_rim),
            imgsz=imgsz,
            device=self.device,
            classes=requested_classes,
            verbose=False,
        )
        if self.use_quantize:
            predict_args["quantize"] = 16
        results = self._model.predict(**predict_args)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return CourtDetections()

        boxes = results[0].boxes
        ball_cands: List[Tuple[Tuple[float, float, float, float], float]] = []
        rim_cands: List[Tuple[Tuple[float, float, float, float], float]] = []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(float, box)
            bbox = (x1, y1, x2, y2)
            radius = max(x2 - x1, y2 - y1) / 2.0
            conf_f = float(conf)

            if cls_id in self._ball_ids:
                if (
                    conf_f >= self.min_confidence
                    and self.min_radius <= radius <= self.max_radius
                ):
                    ball_cands.append((bbox, conf_f))
            elif cls_id in self._rim_ids and conf_f >= self.min_confidence_rim:
                rim_cands.append((bbox, conf_f))

        best_ball = _pick_best(ball_cands)
        best_rim = _pick_best(rim_cands)

        ball = None
        if best_ball is not None:
            (x1, y1, x2, y2), conf = best_ball
            ball = BallDetection(
                center_xy=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=conf,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                measurement_source="yolo",
            )

        rim = None
        if best_rim is not None:
            (x1, y1, x2, y2), conf = best_rim
            rim = RimDetection(
                center_xy=(
                    (x1 + x2) / 2.0,
                    y1 + (y2 - y1) * self.rim_center_y_fraction,
                ),
                bbox_xyxy=(x1, y1, x2, y2),
                confidence=conf,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
            )

        return CourtDetections(ball=ball, rim=rim)

    def _detect_color(
        self,
        bgr_image: np.ndarray,
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallDetection]:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[BallDetection] = None
        best_conf = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * (self.min_radius**2):
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < self.min_circularity:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < self.min_radius or radius > self.max_radius:
                continue
            conf = min(1.0, circularity)
            if conf > best_conf and conf >= self.min_confidence:
                best_conf = conf
                best = BallDetection(
                    center_xy=(float(cx), float(cy)),
                    bbox_xyxy=(
                        float(cx - radius),
                        float(cy - radius),
                        float(cx + radius),
                        float(cy + radius),
                    ),
                    confidence=float(conf),
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    measurement_source="color",
                )
        return best
