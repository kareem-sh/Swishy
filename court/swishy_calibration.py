"""Public meter-based Swichy court calibration API."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from court.frame import KaliCalibCourtAdapter
from court.models import (
    CourtCalibration,
    CourtCoordinateDiagnostic,
    CourtFrame,
    CourtPoint2D,
    CourtPoint3D,
    CourtSanity,
    FibaCourtGeometry,
    SwishyCourtCalibration,
    NATIVE_FIELD_LENGTH,
    NATIVE_FIELD_WIDTH,
)
from utils.config_loader import load_yaml


def adapt_kalicalib_calibration(
    native: CourtCalibration,
    *,
    reference_hoop: str = "near",
    geometry: FibaCourtGeometry | None = None,
    minimum_inliers: int = 4,
    maximum_reprojection_error_px: float = 40.0,
    length_tolerance_m: float = 1.0,
    width_tolerance_m: float = 0.75,
) -> "SwishyCalibrationService":
    """Create Swichy's public CourtFrame service from a native calibration."""
    geometry = geometry or FibaCourtGeometry()
    adapter = KaliCalibCourtAdapter(geometry, reference_hoop)
    sanity = _sanity(native, adapter, minimum_inliers, maximum_reprojection_error_px, length_tolerance_m, width_tolerance_m)
    status = _status(native, sanity)
    message = "Swichy CourtFrame is ready in metres." if status == "VALID" else "; ".join(sanity.failures) or native.message
    confidence = _confidence(native, maximum_reprojection_error_px)
    result = SwishyCourtCalibration(
        native_kalicalib_calibration=native,
        court_frame=CourtFrame(reference_hoop, CourtPoint3D(0.0, 0.0, 0.0)),
        geometry=geometry,
        confidence=confidence,
        sanity=sanity,
        status=status,
        message=message,
        _adapter=adapter,
    )
    return SwishyCalibrationService(result, adapter)


def adapt_from_config(native: CourtCalibration) -> "SwishyCalibrationService":
    """Adapt a native result using the single existing Swichy YAML config."""
    cfg = load_yaml("court.yaml")
    fiba = cfg.get("fiba", {})
    sanity = cfg.get("sanity", {})
    return adapt_kalicalib_calibration(
        native,
        reference_hoop=str(cfg.get("reference_hoop", "near")),
        geometry=FibaCourtGeometry(
            length_m=float(fiba.get("length_m", 28.0)),
            width_m=float(fiba.get("width_m", 15.0)),
            hoop_height_m=float(fiba.get("hoop_height_m", 3.05)),
        ),
        minimum_inliers=int(sanity.get("minimum_inliers", 4)),
        maximum_reprojection_error_px=float(sanity.get("maximum_reprojection_error_px", 40.0)),
        length_tolerance_m=float(sanity.get("length_tolerance_m", 1.0)),
        width_tolerance_m=float(sanity.get("width_tolerance_m", 0.75)),
    )


class SwishyCalibrationService:
    """Coordinate conversion API that hides all KaliCalib-native values."""

    def __init__(self, calibration: SwishyCourtCalibration, adapter: KaliCalibCourtAdapter):
        self.calibration = calibration
        self._adapter = adapter

    def image_to_court(self, point_xy: Sequence[float]) -> CourtPoint3D:
        """Compatibility facade for the canonical 3D CourtFrame conversion."""
        return self.calibration.image_to_court(point_xy)

    def image_to_court_floor(self, point_xy: Sequence[float]) -> CourtPoint3D:
        """Convert an image point known to lie on the floor to ``(X, Y, 0)``."""
        return self.calibration.image_to_court(point_xy)

    def court_to_image(self, point: CourtPoint2D | Sequence[float]) -> tuple[float, float]:
        return self.calibration.court_to_image(point)

    def reference_hoop_3d(self) -> CourtPoint3D:
        return self.calibration.reference_hoop_3d()

    def is_valid(self) -> bool:
        return self.calibration.is_valid()

    def diagnose_image_points(self, points_xy: Sequence[Sequence[float]], round_trip_tolerance_px: float = 1.0) -> list[CourtCoordinateDiagnostic]:
        """Return explicit image -> CourtFrame -> image diagnostics for floor pixels."""
        diagnostics: list[CourtCoordinateDiagnostic] = []
        for raw_point in points_xy:
            image_xy = (float(raw_point[0]), float(raw_point[1]))
            try:
                court = self.calibration.image_to_court(image_xy)
                projected_back = self.calibration.court_to_image(court)
                error = float(np.linalg.norm(np.asarray(projected_back) - np.asarray(image_xy)))
                status = "VALID" if error <= round_trip_tolerance_px else "LOW_CONFIDENCE"
                message = "" if status == "VALID" else f"round-trip error exceeds {round_trip_tolerance_px:.2f} px"
                diagnostics.append(CourtCoordinateDiagnostic(image_xy, court, projected_back, error, status, message))
            except (RuntimeError, ValueError) as exc:
                diagnostics.append(CourtCoordinateDiagnostic(image_xy, None, None, None, self.calibration.status, str(exc)))
        return diagnostics


def _sanity(native: CourtCalibration, adapter: KaliCalibCourtAdapter, minimum_inliers: int, maximum_error: float, length_tolerance: float, width_tolerance: float) -> CourtSanity:
    length = NATIVE_FIELD_LENGTH / adapter.native_units_per_meter_length
    width = NATIVE_FIELD_WIDTH / adapter.native_units_per_meter_width
    failures: list[str] = []
    if not native.success:
        failures.append(f"native calibration failed: {native.status}")
    if native.inlier_count < minimum_inliers:
        failures.append(f"inliers {native.inlier_count} below minimum {minimum_inliers}")
    if native.reprojection_error_px is None or native.reprojection_error_px > maximum_error:
        failures.append(f"reprojection error exceeds {maximum_error:.1f} px")
    if abs(length - adapter.geometry.length_m) > length_tolerance:
        failures.append("estimated court length is outside tolerance")
    if abs(width - adapter.geometry.width_m) > width_tolerance:
        failures.append("estimated court width is outside tolerance")
    return CourtSanity(
        length, width, adapter.reference_hoop, native.inlier_count,
        native.reprojection_error_px, not failures, tuple(failures),
        abs(length - adapter.geometry.length_m),
        abs(width - adapter.geometry.width_m),
    )


def _confidence(native: CourtCalibration, maximum_error: float) -> float:
    if not native.success or native.reprojection_error_px is None:
        return 0.0
    inlier_fraction = min(native.inlier_count / max(len(native.detection.valid_keypoints), 1), 1.0)
    error_quality = max(0.0, 1.0 - native.reprojection_error_px / maximum_error)
    return round(inlier_fraction * error_quality, 3)


def _status(native: CourtCalibration, sanity: CourtSanity) -> str:
    if not native.success or native.homography_image_to_court is None or native.homography_court_to_image is None:
        return "INVALID"
    return "VALID" if sanity.passed else "LOW_CONFIDENCE"
