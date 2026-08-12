"""Tests for the observed ball trajectory overlay recorder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ball.models import BallSnapshot
from ball.trajectory_overlay import ObservedTrajectoryRecorder


def _ball(
    x: float,
    y: float,
    frame: int = 0,
    *,
    interpolated: bool = False,
) -> BallSnapshot:
    return BallSnapshot(
        center_xy=(x, y),
        confidence=0.9,
        frame_index=frame,
        timestamp_ms=frame * 33,
        is_interpolated=interpolated,
    )


def test_path_follows_current_rim_after_camera_motion() -> None:
    recorder = ObservedTrajectoryRecorder()
    recorder.update(
        _ball(100, 100),
        released_this_frame=True,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )
    recorder.update(
        _ball(130, 85, frame=1),
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(210, 105),
        rim_radius=20,
    )

    segments = recorder.screen_segments((210, 105), 20)
    assert segments == [[(110.0, 105.0), (130.0, 85.0)]]


def test_missing_detection_starts_a_new_segment() -> None:
    recorder = ObservedTrajectoryRecorder()
    recorder.update(
        _ball(100, 100),
        released_this_frame=True,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )
    recorder.update(
        None,
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )
    recorder.update(
        _ball(140, 80, frame=2),
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )

    segments = recorder.screen_segments((200, 100), 20)
    assert len(segments) == 2
    assert segments[0] == [(100.0, 100.0)]
    assert segments[1] == [(140.0, 80.0)]


def test_finished_shot_stops_recording_until_next_release() -> None:
    recorder = ObservedTrajectoryRecorder()
    recorder.update(
        _ball(100, 100),
        released_this_frame=True,
        shot_finished=True,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )
    recorder.update(
        _ball(130, 80, frame=1),
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )

    assert not recorder.active
    assert recorder.screen_segments((200, 100), 20) == [[(100.0, 100.0)]]


def test_interpolated_snapshot_is_not_recorded() -> None:
    recorder = ObservedTrajectoryRecorder()
    recorder.update(
        _ball(100, 100),
        released_this_frame=True,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )
    recorder.update(
        _ball(120, 80, frame=1, interpolated=True),
        released_this_frame=False,
        shot_finished=False,
        rim_center_xy=(200, 100),
        rim_radius=20,
    )

    assert recorder.screen_segments((200, 100), 20) == [[(100.0, 100.0)]]


def test_robust_fit_ignores_one_jitter_outlier() -> None:
    recorder = ObservedTrajectoryRecorder(
        maximum_jump_rim_radii=100,
        fit_outlier_threshold_rim_radii=0.5,
    )
    for frame in range(9):
        # Smooth projectile-like samples, with one deliberately bad center.
        relative_x = -8.0 + frame
        relative_y = 0.12 * (frame - 4) ** 2 - 2.0
        if frame == 4:
            relative_y += 5.0
        recorder.update(
            _ball(
                500 + relative_x * 20,
                300 + relative_y * 20,
                frame=frame,
            ),
            released_this_frame=frame == 0,
            shot_finished=False,
            rim_center_xy=(500, 300),
            rim_radius=20,
        )

    fitted = recorder.fitted_screen_curve((500, 300), 20)
    assert len(fitted) == 60
    middle_y = fitted[len(fitted) // 2][1]
    assert middle_y < 280.0


if __name__ == "__main__":
    test_path_follows_current_rim_after_camera_motion()
    test_missing_detection_starts_a_new_segment()
    test_finished_shot_stops_recording_until_next_release()
    test_interpolated_snapshot_is_not_recorded()
    test_robust_fit_ignores_one_jitter_outlier()
    print("All observed trajectory overlay tests passed.")
