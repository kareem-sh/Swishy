"""The explicit unit and orientation adapter from KaliCalib to Swichy metres."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from court.models import CourtPoint2D, FibaCourtGeometry, NATIVE_FIELD_LENGTH, NATIVE_FIELD_WIDTH
from court.units import court_to_native_xy, native_to_court_xy


@dataclass(frozen=True)
class KaliCalibCourtAdapter:
    """Convert native KaliCalib court-plane values to a Swichy CourtFrame.

    KaliCalib's X axis follows the 2800-unit court length and its Y axis follows
    the 1500-unit width. One metre is derived once here from the authoritative
    FIBA dimensions; no callers perform their own unit conversion.
    """

    geometry: FibaCourtGeometry
    reference_hoop: str
    near_hoop_native_xy: tuple[float, float] = (157.5, 750.0)
    far_hoop_native_xy: tuple[float, float] = (2642.5, 750.0)

    def __post_init__(self) -> None:
        if self.reference_hoop not in {"near", "far"}:
            raise ValueError("reference_hoop must be either 'near' or 'far'.")

    @property
    def native_units_per_meter_length(self) -> float:
        return NATIVE_FIELD_LENGTH / self.geometry.length_m

    @property
    def native_units_per_meter_width(self) -> float:
        return NATIVE_FIELD_WIDTH / self.geometry.width_m

    @property
    def reference_hoop_native_xy(self) -> tuple[float, float]:
        return self.near_hoop_native_xy if self.reference_hoop == "near" else self.far_hoop_native_xy

    def native_to_court(self, native_xy: tuple[float, float] | np.ndarray) -> CourtPoint2D:
        x_m, y_m = native_to_court_xy(native_xy, self.geometry, self.reference_hoop, hoop_xy=self.reference_hoop_native_xy)
        return CourtPoint2D(x_m=x_m, y_m=y_m)

    def court_to_native(self, point: CourtPoint2D) -> tuple[float, float]:
        return court_to_native_xy((point.x_m, point.y_m), self.geometry, self.reference_hoop, hoop_xy=self.reference_hoop_native_xy)
