"""Isolated KaliCalib court-calibration subsystem."""

from court.calibration import CourtCalibrator, draw_calibration_debug, get_field_points
from court.models import (
    CameraModel, CourtCalibration, CourtCoordinateDiagnostic, CourtDetection, CourtFrame, CourtKeypoint,
    CourtPoint2D, CourtPoint3D, CourtSanity, FibaCourtGeometry,
    SwishyCourtCalibration,
)
from court.swishy_calibration import SwishyCalibrationService, adapt_from_config, adapt_kalicalib_calibration
from court.units import court_to_native_xy, native_to_court_xy, native_units_per_meter

__all__ = [
    "CameraModel",
    "CourtCalibration",
    "CourtCalibrator",
    "CourtDetection",
    "CourtCoordinateDiagnostic",
    "CourtDetector",
    "CourtKeypoint",
    "CourtPoint2D",
    "CourtPoint3D",
    "CourtSanity",
    "CourtFrame",
    "FibaCourtGeometry",
    "SwishyCourtCalibration",
    "SwishyCalibrationService",
    "adapt_from_config",
    "adapt_kalicalib_calibration",
    "court_to_native_xy",
    "draw_calibration_debug",
    "get_field_points",
    "native_to_court_xy",
    "native_units_per_meter",
]


def __getattr__(name: str):
    """Avoid importing the CLI module before ``python -m court.detector`` runs."""
    if name == "CourtDetector":
        from court.detector import CourtDetector

        return CourtDetector
    raise AttributeError(name)
