"""Typed public results for the isolated KaliCalib integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


NATIVE_FIELD_LENGTH = 2800.0
NATIVE_FIELD_WIDTH = 1500.0
NATIVE_UNIT = "KaliCalib native court units"


@dataclass(frozen=True)
class CourtPoint2D:
    """A point on the Swichy court floor, always in metres."""

    x_m: float
    y_m: float


@dataclass(frozen=True)
class CourtPoint3D:
    """A point in the Swichy CourtFrame, always in metres."""

    x_m: float
    y_m: float
    z_m: float


@dataclass(frozen=True)
class FibaCourtGeometry:
    """The single authoritative FIBA court geometry used by Swichy."""

    length_m: float = 28.0
    width_m: float = 15.0
    hoop_height_m: float = 3.05


@dataclass(frozen=True)
class CourtFrame:
    """Swichy's global floor coordinate frame, independent of PoseWorldFrame.

    Origin is the selected hoop's floor projection. +Y points from that hoop
    toward the opposite baseline; +X is court-right while looking along +Y.
    +Z is vertical. Coordinates returned through this frame are metres.
    """

    reference_hoop: str
    origin: CourtPoint3D
    x_axis: str = "court-right looking from reference hoop toward opposite baseline"
    y_axis: str = "reference hoop toward opposite baseline"
    z_axis: str = "vertically upward"


@dataclass(frozen=True)
class CourtSanity:
    """Quality checks for a converted meter-based court calibration."""

    estimated_court_length_m: float
    estimated_court_width_m: float
    orientation: str
    inlier_count: int
    reprojection_error_px: Optional[float]
    passed: bool
    failures: tuple[str, ...] = ()
    length_error_m: float = 0.0
    width_error_m: float = 0.0


@dataclass(frozen=True)
class CourtCoordinateDiagnostic:
    """One verified image-floor-pixel <-> CourtFrame conversion."""

    image_xy: tuple[float, float]
    court_point: Optional[CourtPoint3D]
    projected_back_xy: Optional[tuple[float, float]]
    round_trip_error_px: Optional[float]
    status: str
    message: str = ""


@dataclass(frozen=True)
class CourtKeypoint:
    """One KaliCalib keypoint and its matching native court-plane point."""

    index: int
    image_xy: tuple[float, float]
    court_xy: tuple[float, float]
    confidence: float


@dataclass
class CourtDetection:
    """Keypoints predicted by KaliCalib for one input image."""

    image_width: int
    image_height: int
    keypoints: list[CourtKeypoint] = field(default_factory=list)
    valid_keypoints: list[CourtKeypoint] = field(default_factory=list)
    inlier_keypoints: list[CourtKeypoint] = field(default_factory=list)
    model_confidence: Optional[float] = None
    status: str = "ok"
    message: str = ""
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera estimate compatible with KaliCalib's planar points."""

    intrinsic_matrix: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    distortion_coefficients: np.ndarray


@dataclass
class CourtCalibration:
    """A calibrated image in native KaliCalib court-plane coordinates."""

    detection: CourtDetection
    homography_image_to_court: Optional[np.ndarray]
    homography_court_to_image: Optional[np.ndarray]
    camera_model: Optional[CameraModel]
    reprojection_error_px: Optional[float]
    inlier_count: int
    status: str
    message: str
    image_width: int
    image_height: int
    reference_geometry: dict[str, Any]
    coordinate_unit: str = NATIVE_UNIT

    @property
    def success(self) -> bool:
        return self.status == "ok"


@dataclass
class SwishyCourtCalibration:
    """Meter-based adapter around an immutable native KaliCalib calibration."""

    native_kalicalib_calibration: CourtCalibration
    court_frame: CourtFrame
    geometry: FibaCourtGeometry
    confidence: float
    sanity: CourtSanity
    status: str
    message: str
    _adapter: Any = field(repr=False, compare=False, default=None)

    @property
    def reprojection_error_px(self) -> Optional[float]:
        return self.native_kalicalib_calibration.reprojection_error_px

    @property
    def inlier_count(self) -> int:
        return self.native_kalicalib_calibration.inlier_count

    @property
    def image_width(self) -> int:
        return self.native_kalicalib_calibration.image_width

    @property
    def image_height(self) -> int:
        return self.native_kalicalib_calibration.image_height

    def is_valid(self) -> bool:
        return self.status == "VALID" and self.sanity.passed

    def image_to_court(self, point_xy: tuple[float, float] | list[float]) -> CourtPoint3D:
        """Map a known floor pixel to ``(X, Y, 0)`` in CourtFrame metres.

        Coordinates are withheld unless the calibration passes its configured
        quality gates; this prevents callers from treating a weak homography as
        a trusted player/court location.
        """
        if not self.is_valid():
            raise RuntimeError(f"Court calibration is {self.status}: {self.message}")
        if self._adapter is None or self.native_kalicalib_calibration.homography_image_to_court is None:
            raise RuntimeError("Court calibration has no image-to-court transform.")
        import cv2

        native = cv2.perspectiveTransform(
            np.asarray(point_xy, dtype=np.float32).reshape(1, 1, 2),
            self.native_kalicalib_calibration.homography_image_to_court,
        )[0, 0]
        point = self._adapter.native_to_court(native)
        return CourtPoint3D(point.x_m, point.y_m, 0.0)

    def court_to_image(self, point: CourtPoint2D | CourtPoint3D | tuple[float, float] | list[float]) -> tuple[float, float]:
        """Project a CourtFrame floor point in metres back into image pixels."""
        if not self.is_valid():
            raise RuntimeError(f"Court calibration is {self.status}: {self.message}")
        if self._adapter is None or self.native_kalicalib_calibration.homography_court_to_image is None:
            raise RuntimeError("Court calibration has no court-to-image transform.")
        import cv2

        if isinstance(point, CourtPoint3D):
            if point.z_m != 0.0:
                raise ValueError("court_to_image accepts floor points only (z_m must be 0).")
            floor_point = CourtPoint2D(point.x_m, point.y_m)
        elif isinstance(point, CourtPoint2D):
            floor_point = point
        else:
            floor_point = CourtPoint2D(float(point[0]), float(point[1]))
        native = self._adapter.court_to_native(floor_point)
        image = cv2.perspectiveTransform(
            np.asarray(native, dtype=np.float32).reshape(1, 1, 2),
            self.native_kalicalib_calibration.homography_court_to_image,
        )[0, 0]
        return float(image[0]), float(image[1])

    def reference_hoop_3d(self) -> CourtPoint3D:
        return CourtPoint3D(0.0, 0.0, self.geometry.hoop_height_m)
