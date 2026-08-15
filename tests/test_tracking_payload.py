"""Unit tests for frontend tracking payload serialization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallHolder, ShooterCourtPosition
from feedback.tracking_payload import (
    ball_holder_to_dict,
    frame_tracking_to_dict,
    shooter_release_to_dict,
    write_releases_csv,
)
from pipeline import FrameResult


def test_ball_holder_payload_contains_court_coordinates():
    holder = BallHolder(
        player_id=2,
        confidence=0.91,
        image_position=(640.0, 480.0),
        court_position=(2.5, 7.8, 0.0),
        tracking_status="CONFIDENT",
        shooter_state="CONFIRMED_SHOOTER",
        distance_to_hoop_m=8.2,
    )
    payload = ball_holder_to_dict(holder)
    assert payload["player_id"] == 2
    assert payload["court_x_m"] == 2.5
    assert payload["court_y_m"] == 7.8
    assert payload["distance_to_hoop_m"] == 8.2


def test_frame_tracking_payload_includes_court_state():
    frame_result = FrameResult(
        timestamp_ms=1000,
        court_calibration_state="VALID",
        court_calibration_valid=True,
        court_inlier_count=32,
        court_reprojection_error_px=18.2,
        selected_pose_index=1,
        pose_candidate_count=4,
        ball_holder=BallHolder(player_id=1, confidence=0.8),
    )
    payload = frame_tracking_to_dict(10, frame_result)
    assert payload["frame_index"] == 10
    assert payload["court"]["state"] == "VALID"
    assert payload["court"]["geometry"]["standard"] == "FIBA"
    assert payload["ball_holder"]["player_id"] == 1


def test_releases_csv_round_trip(tmp_path):
    release = ShooterCourtPosition(
        shooter_id=3,
        release_frame=120,
        release_timestamp_ms=4000,
        confidence=0.92,
        image_x_px=512.0,
        image_y_px=640.0,
        court_x_m=2.74,
        court_y_m=7.83,
        distance_to_hoop_m=8.30,
        status="VALID",
        court_calibration_valid=True,
        court_reprojection_error_px=18.0,
        court_inliers=30,
    )
    csv_path = write_releases_csv([release], tmp_path / "releases.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "shooter_court_x_m" in text
    assert "2.74" in text
    assert shooter_release_to_dict(release)["status"] == "VALID"
