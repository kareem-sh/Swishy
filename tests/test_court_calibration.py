"""Pure unit tests for the isolated KaliCalib court subsystem."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from court.calibration import CourtCalibrator, get_field_points, reference_geometry
from court.detector import CourtDetector, resolve_device, validate_model_path
from court.models import CourtCalibration, CourtDetection, CourtKeypoint
from court.swishy_calibration import adapt_kalicalib_calibration


def _detection(points):
    return CourtDetection(640, 360, keypoints=points, valid_keypoints=points)


def test_model_path_validation_and_missing_model_behavior():
    assert validate_model_path("models/court/model_challenge.pth").is_file()
    try:
        validate_model_path("models/court/not-present.pth")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing KaliCalib weights must be explicit")


def test_cpu_device_is_supported():
    assert resolve_device("cpu") == "cpu"


def test_invalid_image_returns_structured_failure_without_inference():
    detector = object.__new__(CourtDetector)
    result = detector.detect(None)
    assert result.status == "invalid_image"
    assert not result.success


def test_homography_and_successful_result_creation():
    court = get_field_points()[:4, :2]
    image = [(50.0, 40.0), (150.0, 40.0), (250.0, 40.0), (350.0, 40.0)]
    # Use non-collinear official points from rows 0, 1, 13, 14.
    indices = [0, 1, 13, 14]
    court = get_field_points()[indices, :2]
    image = [(50.0, 40.0), (250.0, 60.0), (60.0, 180.0), (260.0, 200.0)]
    points = [CourtKeypoint(i, xy, tuple(court[n]), 0.9) for n, (i, xy) in enumerate(zip(indices, image))]
    calibration = CourtCalibrator().calibrate(np.zeros((360, 640, 3), dtype=np.uint8), _detection(points))
    assert calibration.success
    assert calibration.homography_image_to_court.shape == (3, 3)
    assert calibration.homography_court_to_image.shape == (3, 3)
    assert calibration.inlier_count == 4


def test_insufficient_points_fails_gracefully():
    points = [CourtKeypoint(i, (float(i), float(i)), (float(i), float(i + 1)), 0.9) for i in range(3)]
    calibration = CourtCalibrator().calibrate(np.zeros((10, 10, 3), dtype=np.uint8), _detection(points))
    assert calibration.status == "insufficient_points"
    assert not calibration.success


def _native_calibration_for_round_trip():
    # Native KaliCalib plane -> image is affine here so geometry tests remain
    # pure and do not need the neural model or a GPU.
    court_to_image = np.array([[0.20, 0.03, 50.0], [0.01, 0.16, 40.0], [0.0, 0.0, 1.0]])
    image_to_court = np.linalg.inv(court_to_image)
    detection = CourtDetection(640, 360)
    return CourtCalibration(
        detection=detection,
        homography_image_to_court=image_to_court,
        homography_court_to_image=court_to_image,
        camera_model=None,
        reprojection_error_px=4.0,
        inlier_count=20,
        status="ok",
        message="synthetic",
        image_width=640,
        image_height=360,
        reference_geometry=reference_geometry(),
    )


def test_meter_court_frame_round_trip_for_required_floor_points():
    service = adapt_kalicalib_calibration(_native_calibration_for_round_trip(), reference_hoop="near")
    assert service.is_valid()
    assert service.calibration.sanity.length_error_m == 0.0
    assert service.calibration.sanity.width_error_m == 0.0
    # Hoop, center, left/right boundaries, and both endline directions.
    points = [(0.0, 0.0), (0.0, 14.0), (0.0, 28.0), (-7.5, 14.0), (7.5, 14.0), (0.0, -1.575), (0.0, 26.425)]
    for expected in points:
        image = service.court_to_image(expected)
        actual = service.image_to_court_floor(image)
        assert abs(actual.x_m - expected[0]) < 1e-4
        assert abs(actual.y_m - expected[1]) < 1e-4
        assert actual.z_m == 0.0
    hoop = service.reference_hoop_3d()
    assert (hoop.x_m, hoop.y_m, hoop.z_m) == (0.0, 0.0, 3.05)


def test_far_reference_reverses_court_orientation_without_changing_metres():
    service = adapt_kalicalib_calibration(_native_calibration_for_round_trip(), reference_hoop="far")
    assert service.calibration.court_frame.reference_hoop == "far"
    point = service.image_to_court(service.court_to_image((3.0, 6.0)))
    assert abs(point.x_m - 3.0) < 1e-4
    assert abs(point.y_m - 6.0) < 1e-4


def test_axis_directions_are_explicit_in_the_adapter_layer():
    service = adapt_kalicalib_calibration(_native_calibration_for_round_trip(), reference_hoop="near")
    adapter = service.calibration._adapter
    assert adapter.native_to_court((157.5, 750.0)).x_m == 0.0
    assert adapter.native_to_court((157.5, 750.0)).y_m == 0.0
    assert adapter.native_to_court((157.5, 850.0)).x_m > 0.0  # court-right
    assert adapter.native_to_court((257.5, 750.0)).y_m > 0.0  # away from hoop


def test_low_confidence_calibration_refuses_trusted_coordinates():
    native = replace(_native_calibration_for_round_trip(), reprojection_error_px=100.0)
    service = adapt_kalicalib_calibration(native, maximum_reprojection_error_px=40.0)
    assert service.calibration.status == "LOW_CONFIDENCE"
    assert not service.is_valid()
    try:
        service.image_to_court((100.0, 100.0))
    except RuntimeError as exc:
        assert "LOW_CONFIDENCE" in str(exc)
    else:
        raise AssertionError("Low-confidence calibration must not return trusted coordinates")


def test_diagnostics_report_a_precise_round_trip():
    service = adapt_kalicalib_calibration(_native_calibration_for_round_trip())
    diagnostics = service.diagnose_image_points([(100.0, 100.0)], round_trip_tolerance_px=0.01)
    assert len(diagnostics) == 1
    assert diagnostics[0].status == "VALID"
    assert diagnostics[0].court_point is not None
    assert diagnostics[0].court_point.z_m == 0.0
    assert diagnostics[0].round_trip_error_px < 0.01


if __name__ == "__main__":
    test_model_path_validation_and_missing_model_behavior()
    test_cpu_device_is_supported()
    test_invalid_image_returns_structured_failure_without_inference()
    test_homography_and_successful_result_creation()
    test_insufficient_points_fails_gracefully()
    test_meter_court_frame_round_trip_for_required_floor_points()
    test_far_reference_reverses_court_orientation_without_changing_metres()
    test_axis_directions_are_explicit_in_the_adapter_layer()
    test_low_confidence_calibration_refuses_trusted_coordinates()
    test_diagnostics_report_a_precise_round_trip()
    print("Court calibration tests passed.")
