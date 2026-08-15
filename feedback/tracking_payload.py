"""JSON payloads for ball-holder and court tracking consumed by the frontend."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ball.models import BallHolder, ShooterCourtPosition
    from pipeline import FrameResult, ShotAnalysisPipeline


def ball_holder_to_dict(holder: Optional["BallHolder"]) -> Optional[dict]:
    if holder is None:
        return None
    court_x = court_y = court_z = None
    if holder.court_position is not None:
        court_x, court_y, court_z = holder.court_position
    image_x = image_y = None
    if holder.image_position is not None:
        image_x, image_y = holder.image_position
    return {
        "player_id": holder.player_id,
        "confidence": round(holder.confidence, 3),
        "tracking_status": holder.tracking_status,
        "shooter_state": holder.shooter_state,
        "image_x_px": None if image_x is None else round(image_x, 1),
        "image_y_px": None if image_y is None else round(image_y, 1),
        "court_x_m": None if court_x is None else round(court_x, 2),
        "court_y_m": None if court_y is None else round(court_y, 2),
        "court_z_m": None if court_z is None else round(court_z, 2),
        "distance_to_hoop_m": (
            None if holder.distance_to_hoop_m is None else round(holder.distance_to_hoop_m, 2)
        ),
    }


def court_state_to_dict(frame_result: "FrameResult") -> dict:
    return {
        "state": frame_result.court_calibration_state,
        "valid": frame_result.court_calibration_valid,
        "confidence": frame_result.court_calibration_confidence,
        "reprojection_error_px": frame_result.court_reprojection_error_px,
        "inliers": frame_result.court_inlier_count,
        "reference_hoop": frame_result.court_reference_hoop,
        "last_failure": frame_result.court_last_failure,
        "geometry": {
            "length_m": 28.0,
            "width_m": 15.0,
            "hoop_height_m": 3.05,
            "standard": "FIBA",
        },
    }


def shooter_release_to_dict(release: "ShooterCourtPosition") -> dict:
    return {
        "shot_id": release.release_frame,
        "release_frame": release.release_frame,
        "release_timestamp_ms": release.release_timestamp_ms,
        "shooter_id": release.shooter_id,
        "shooter_confidence": round(release.confidence, 3),
        "shooter_image_x": release.image_x_px,
        "shooter_image_y": release.image_y_px,
        "shooter_court_x_m": release.court_x_m,
        "shooter_court_y_m": release.court_y_m,
        "shooter_court_z_m": release.court_z_m,
        "distance_to_hoop_m": release.distance_to_hoop_m,
        "court_calibration_valid": release.court_calibration_valid,
        "court_reprojection_error_px": release.court_reprojection_error_px,
        "court_inliers": release.court_inliers,
        "status": release.status,
    }


def frame_tracking_to_dict(frame_index: int, frame_result: "FrameResult") -> dict:
    ball_xy = None
    if frame_result.ball is not None:
        ball_xy = {
            "x_px": round(frame_result.ball.x, 1),
            "y_px": round(frame_result.ball.y, 1),
            "confidence": round(frame_result.ball.confidence, 3),
        }
    elif frame_result.ball_snapshot is not None:
        ball_xy = {
            "x_px": round(frame_result.ball_snapshot.x, 1),
            "y_px": round(frame_result.ball_snapshot.y, 1),
            "confidence": round(frame_result.ball_snapshot.confidence, 3),
        }
    return {
        "frame_index": frame_index,
        "timestamp_ms": frame_result.timestamp_ms,
        "ball": ball_xy,
        "ball_state": frame_result.ball_state,
        "selected_pose_index": frame_result.selected_pose_index,
        "pose_candidate_count": frame_result.pose_candidate_count,
        "ball_holder": ball_holder_to_dict(frame_result.ball_holder),
        "court": court_state_to_dict(frame_result),
        "release": (
            shooter_release_to_dict(frame_result.shooter_release)
            if frame_result.shooter_release is not None
            else None
        ),
    }


def session_tracking_to_dict(
    pipeline: "ShotAnalysisPipeline",
    frame_records: List[dict],
) -> dict:
    calibration = pipeline.court_service
    session_court = {
        "state": pipeline.court_calibration_state,
        "valid": calibration is not None and pipeline.court_calibration_state == "VALID",
        "reference_hoop": (
            calibration.calibration.court_frame.reference_hoop
            if calibration is not None
            else None
        ),
        "geometry": {
            "length_m": 28.0,
            "width_m": 15.0,
            "hoop_height_m": 3.05,
            "standard": "FIBA",
        },
    }
    if calibration is not None:
        native = pipeline.court_native_calibration
        session_court.update(
            {
                "confidence": calibration.calibration.confidence,
                "reprojection_error_px": (
                    None if native is None else native.reprojection_error_px
                ),
                "inliers": None if native is None else native.inlier_count,
            }
        )

    releases = [
        shooter_release_to_dict(record) for record in pipeline.release_records
    ]
    return {
        "court": session_court,
        "frames": frame_records,
        "releases": releases,
        "shooter_switches": pipeline.shooter_switches,
    }


def write_releases_csv(releases: List["ShooterCourtPosition"], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shot_id",
        "release_frame",
        "release_timestamp_ms",
        "shooter_id",
        "shooter_confidence",
        "shooter_image_x",
        "shooter_image_y",
        "shooter_court_x_m",
        "shooter_court_y_m",
        "shooter_court_z_m",
        "distance_to_hoop_m",
        "court_calibration_valid",
        "court_reprojection_error_px",
        "court_inliers",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for release in releases:
            writer.writerow(shooter_release_to_dict(release))
    return output_path
