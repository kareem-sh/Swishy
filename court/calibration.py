"""KaliCalib-compatible court geometry and calibration estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from court.models import (
    CameraModel,
    CourtCalibration,
    CourtDetection,
    CourtKeypoint,
    NATIVE_FIELD_LENGTH,
    NATIVE_FIELD_WIDTH,
    NATIVE_UNIT,
)


def get_field_points() -> np.ndarray:
    """Return KaliCalib's 93 predefined 3D court points, unchanged.

    The first 91 points are the heatmap channels used for homography. The last
    two are basket centres, as defined by the official challenge implementation.
    """
    points: list[list[float]] = []
    u, s = 175.0, 0.0
    for _ in range(7):
        for i in range(13):
            points.append([i * NATIVE_FIELD_LENGTH / 12.0, NATIVE_FIELD_WIDTH - s, 0.0])
        s += u
        u += 30.0
    basket_x_shift = 120.0 + 15.0 + 45.0 / 2.0
    points.extend([
        [basket_x_shift, NATIVE_FIELD_WIDTH / 2.0, -305.0],
        [NATIVE_FIELD_LENGTH - basket_x_shift, NATIVE_FIELD_WIDTH / 2.0, -305.0],
    ])
    return np.asarray(points, dtype=np.float64)


def reference_geometry() -> dict[str, object]:
    return {
        "field_length": NATIVE_FIELD_LENGTH,
        "field_width": NATIVE_FIELD_WIDTH,
        "unit": NATIVE_UNIT,
        "point_count": int(len(get_field_points())),
        "heatmap_keypoint_count": 91,
    }


class CourtCalibrator:
    """Estimate a court-plane homography and optional pinhole camera model."""

    def calibrate(self, frame: np.ndarray, detection: CourtDetection) -> CourtCalibration:
        height, width = _image_size(frame, detection)
        base = dict(
            detection=detection,
            image_width=width,
            image_height=height,
            reference_geometry=reference_geometry(),
            coordinate_unit=NATIVE_UNIT,
        )
        if not detection.success:
            return CourtCalibration(
                homography_image_to_court=None, homography_court_to_image=None,
                camera_model=None, reprojection_error_px=None, inlier_count=0,
                status="detection_failed", message=detection.message, **base,
            )

        points = detection.valid_keypoints
        if len(points) < 4:
            return CourtCalibration(
                homography_image_to_court=None, homography_court_to_image=None,
                camera_model=None, reprojection_error_px=None, inlier_count=0,
                status="insufficient_points", message="At least four valid court keypoints are required.", **base,
            )

        image_points = np.asarray([p.image_xy for p in points], dtype=np.float64)
        court_points = np.asarray([p.court_xy for p in points], dtype=np.float64)
        court_to_image, inlier_mask = cv2.findHomography(
            court_points, image_points, cv2.RANSAC, ransacReprojThreshold=35.0, maxIters=2000
        )
        if court_to_image is None or inlier_mask is None:
            return CourtCalibration(
                homography_image_to_court=None, homography_court_to_image=None,
                camera_model=None, reprojection_error_px=None, inlier_count=0,
                status="homography_failed", message="OpenCV could not estimate a court homography.", **base,
            )

        image_to_court = np.linalg.inv(court_to_image)
        inliers = inlier_mask.reshape(-1).astype(bool)
        inlier_count = int(inliers.sum())
        detection.inlier_keypoints = [point for point, is_inlier in zip(points, inliers) if is_inlier]
        detection.debug["inlier_indices"] = [point.index for point in detection.inlier_keypoints]
        projected = cv2.perspectiveTransform(court_points[inliers].reshape(1, -1, 2).astype(np.float32), court_to_image)[0]
        reprojection_error = float(np.mean(np.linalg.norm(projected - image_points[inliers], axis=1))) if inlier_count else None
        camera_model = self._estimate_camera(image_points[inliers], court_points[inliers], width, height)
        return CourtCalibration(
            homography_image_to_court=image_to_court,
            homography_court_to_image=court_to_image,
            camera_model=camera_model,
            reprojection_error_px=reprojection_error,
            inlier_count=inlier_count,
            status="ok",
            message="Calibration estimated from KaliCalib keypoints.",
            **base,
        )

    @staticmethod
    def _estimate_camera(image_points: np.ndarray, court_points: np.ndarray, width: int, height: int) -> CameraModel | None:
        # This follows the official compute_camera_model requirement (>5 planar
        # correspondences) without importing its heavyweight deepsport package.
        if len(image_points) <= 5:
            return None
        object_points = np.column_stack([court_points, np.zeros(len(court_points))]).astype(np.float32)
        try:
            _, intrinsic, distortion, rotations, translations = cv2.calibrateCamera(
                [object_points], [image_points.astype(np.float32)], (width, height), None, None,
                flags=(cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST),
            )
        except cv2.error:
            return None
        return CameraModel(intrinsic, rotations[0], translations[0], distortion)


def _image_size(frame: np.ndarray, detection: CourtDetection) -> tuple[int, int]:
    if isinstance(frame, np.ndarray) and frame.ndim >= 2:
        return int(frame.shape[0]), int(frame.shape[1])
    return detection.image_height, detection.image_width


if TYPE_CHECKING:
    from court.models import CourtCoordinateDiagnostic
    from court.swishy_calibration import SwishyCalibrationService


def draw_calibration_debug(
    frame: np.ndarray,
    calibration: CourtCalibration,
    swishy: "SwishyCalibrationService | None" = None,
    diagnostics: Sequence["CourtCoordinateDiagnostic"] = (),
) -> np.ndarray:
    """Overlay all detections, RANSAC inliers, native court, and CourtFrame."""
    canvas = frame.copy()
    # All heatmap detections are green. Draw true RANSAC inliers on top in red.
    for point in calibration.detection.keypoints:
        xy = tuple(round(v) for v in point.image_xy)
        cv2.circle(canvas, xy, 3, (0, 220, 0), -1)
    for point in calibration.detection.inlier_keypoints:
        xy = tuple(round(v) for v in point.image_xy)
        cv2.circle(canvas, xy, 6, (0, 0, 255), 2)
        cv2.putText(canvas, str(point.index), (xy[0] + 6, xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    if calibration.homography_court_to_image is not None:
        outline = np.asarray([[0, 0], [NATIVE_FIELD_LENGTH, 0], [NATIVE_FIELD_LENGTH, NATIVE_FIELD_WIDTH], [0, NATIVE_FIELD_WIDTH]], dtype=np.float32)
        projected = cv2.perspectiveTransform(outline.reshape(1, -1, 2), calibration.homography_court_to_image)[0].astype(np.int32)
        cv2.polylines(canvas, [projected], True, (255, 255, 0), 2, cv2.LINE_AA)
    if swishy is not None and calibration.success:
        _draw_court_frame(canvas, swishy)
    _draw_coordinate_diagnostics(canvas, diagnostics)
    _draw_debug_text(canvas, calibration, swishy)
    return canvas


def _draw_court_frame(canvas: np.ndarray, swishy: "SwishyCalibrationService") -> None:
    origin = swishy.court_to_image((0.0, 0.0))
    x_end = swishy.court_to_image((3.0, 0.0))
    y_end = swishy.court_to_image((0.0, 3.0))
    origin_px = tuple(round(v) for v in origin)
    cv2.circle(canvas, origin_px, 8, (0, 255, 255), -1)
    cv2.putText(canvas, "Origin / hoop", (origin_px[0] + 8, origin_px[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    for endpoint, label, color in ((x_end, "+X", (255, 0, 255)), (y_end, "+Y", (255, 255, 255))):
        end_px = tuple(round(v) for v in endpoint)
        cv2.arrowedLine(canvas, origin_px, end_px, color, 2, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(canvas, label, (end_px[0] + 5, end_px[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_debug_text(canvas: np.ndarray, calibration: CourtCalibration, swishy: "SwishyCalibrationService | None") -> None:
    lines = [
        f"Inliers: {calibration.inlier_count}",
        f"Reprojection error: {calibration.reprojection_error_px if calibration.reprojection_error_px is not None else 'N/A'} px",
    ]
    if swishy is not None:
        geometry = swishy.calibration.geometry
        lines.extend([
            f"Court: {geometry.length_m:.1f}m x {geometry.width_m:.1f}m",
            f"Reference hoop: {swishy.calibration.court_frame.reference_hoop}",
        ])
    max_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0] for text in lines)
    cv2.rectangle(canvas, (5, 5), (max_width + 20, 18 + len(lines) * 23), (20, 20, 20), -1)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (12, 26 + i * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_coordinate_diagnostics(canvas: np.ndarray, diagnostics: Sequence["CourtCoordinateDiagnostic"]) -> None:
    for index, diagnostic in enumerate(diagnostics, 1):
        if diagnostic.court_point is None:
            continue
        pixel = tuple(round(v) for v in diagnostic.image_xy)
        court = diagnostic.court_point
        label = f"P{index}: pixel {pixel} -> court ({court.x_m:.2f}, {court.y_m:.2f}, {court.z_m:.2f}) m"
        cv2.drawMarker(canvas, pixel, (0, 255, 255), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (pixel[0] + 8, pixel[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)
