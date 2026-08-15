"""Single authoritative conversion between KaliCalib native court units and Swishy metres."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from court.models import FibaCourtGeometry, NATIVE_FIELD_LENGTH, NATIVE_FIELD_WIDTH


def native_units_per_meter(geometry: FibaCourtGeometry, *, axis: str = "length") -> float:
    """Return how many native court pixels correspond to one meter for the requested axis."""
    if axis == "length":
        return NATIVE_FIELD_LENGTH / geometry.length_m
    if axis == "width":
        return NATIVE_FIELD_WIDTH / geometry.width_m
    raise ValueError("axis must be either 'length' or 'width'.")


def native_to_court_xy(native_xy: Sequence[float], geometry: FibaCourtGeometry, reference_hoop: str, *, hoop_xy: tuple[float, float] | None = None) -> tuple[float, float]:
    """Convert a KaliCalib (native_x, native_y) floor point into Swishy metres."""
    native_x, native_y = float(native_xy[0]), float(native_xy[1])
    hoop_x, hoop_y = hoop_xy if hoop_xy is not None else _reference_hoop_native_xy(reference_hoop)
    if reference_hoop == "near":
        x_m = (native_y - hoop_y) / native_units_per_meter(geometry, axis="width")
        y_m = (native_x - hoop_x) / native_units_per_meter(geometry, axis="length")
    elif reference_hoop == "far":
        x_m = (hoop_y - native_y) / native_units_per_meter(geometry, axis="width")
        y_m = (hoop_x - native_x) / native_units_per_meter(geometry, axis="length")
    else:
        raise ValueError("reference_hoop must be either 'near' or 'far'.")
    return x_m, y_m


def court_to_native_xy(court_xy: Sequence[float], geometry: FibaCourtGeometry, reference_hoop: str, *, hoop_xy: tuple[float, float] | None = None) -> tuple[float, float]:
    """Convert a Swishy court floor point in metres into KaliCalib native coordinates."""
    court_x, court_y = float(court_xy[0]), float(court_xy[1])
    hoop_x, hoop_y = hoop_xy if hoop_xy is not None else _reference_hoop_native_xy(reference_hoop)
    if reference_hoop == "near":
        native_x = hoop_x + court_y * native_units_per_meter(geometry, axis="length")
        native_y = hoop_y + court_x * native_units_per_meter(geometry, axis="width")
    elif reference_hoop == "far":
        native_x = hoop_x - court_y * native_units_per_meter(geometry, axis="length")
        native_y = hoop_y - court_x * native_units_per_meter(geometry, axis="width")
    else:
        raise ValueError("reference_hoop must be either 'near' or 'far'.")
    return native_x, native_y


def _reference_hoop_native_xy(reference_hoop: str) -> tuple[float, float]:
    if reference_hoop == "near":
        return (157.5, 750.0)
    if reference_hoop == "far":
        return (2642.5, 750.0)
    raise ValueError("reference_hoop must be either 'near' or 'far'.")


def as_float_xy(value: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Normalize a 2D point into a float tuple."""
    if len(value) != 2:
        raise ValueError("Expected a 2D point (x, y).")
    return float(value[0]), float(value[1])
