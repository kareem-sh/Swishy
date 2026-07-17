"""Tests for phase detection and biomechanical rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.engine import BiomechanicsEngine
from angles.calculator import AngleResult
from phase_detection.detector import ShotPhaseDetector
from phase_detection.features import KinematicFeatures, extract_features
from phase_detection.phases import PHASE_ORDER


def _make_features(**kwargs) -> KinematicFeatures:
    defaults = dict(
        wrist_y=0.5, wrist_velocity_y=0.0, ankle_y_avg=0.1, ankle_velocity_y=0.0,
        ankle_baseline_y=0.1, knee_angle=120.0, knee_angle_delta=0.0,
        hip_y_avg=0.6, hip_velocity_y=0.0,         elbow_angle=90.0,
        index_y=0.55,
        index_velocity_y=0.0,
        index_align_angle=170.0,
        shoulder_y=0.8, nose_y=0.9, nose_velocity_y=0.0,
        total_velocity=0.05, shooting_side="right",
    )
    defaults.update(kwargs)
    return KinematicFeatures(**defaults)


def test_phase_transitions():
    det = ShotPhaseDetector()
    assert det.phase == "ready_stance"

    # Need hysteresis_frames (5) consecutive loading signals + min_dwell in ready_stance
    loading = _make_features(hip_velocity_y=-0.05, wrist_y=0.4, shoulder_y=0.8)
    for _ in range(5):
        det.update(loading)
    assert det.phase == "loading"


def test_biomechanics_knee_rule():
    engine = BiomechanicsEngine()
    angles = {
        "right_knee": AngleResult("right_knee", 100.0, True, True),
        "right_hip": AngleResult("right_hip", 160.0, True, True),
        "right_elbow": AngleResult("right_elbow", 90.0, True, True),
        "right_shoulder": AngleResult("right_shoulder", 90.0, True, True),
        "trunk": AngleResult("trunk", 15.0, True, True),
    }
    features = _make_features()
    result = engine.evaluate("loading", angles, features, "right")
    assert result.total_count >= 2
    knee_rule = next(r for r in result.active_rules if r.rule_id == "knee_flexion_loading")
    assert knee_rule.passed


def test_phase_order_complete():
    assert len(PHASE_ORDER) == 8
    assert PHASE_ORDER[0] == "ready_stance"
    assert PHASE_ORDER[-1] == "landing"


def test_ready_stance_to_loading_on_wrist_rise():
    det = ShotPhaseDetector()
    still = _make_features(total_velocity=0.05, wrist_y=0.02, hip_y_avg=0.0, shoulder_y=-0.4)
    for _ in range(3):
        det.update(still)

    rising = _make_features(
        total_velocity=0.08,
        wrist_y=0.03,
        wrist_velocity_y=0.05,
        hip_y_avg=0.0,
        shoulder_y=-0.4,
    )
    for _ in range(5):
        det.update(rising)
    assert det.phase == "loading"


def test_index_tip_can_confirm_release():
    det = ShotPhaseDetector()
    det.phase = "ball_lift"
    index_release = _make_features(
        wrist_y=0.6,
        wrist_velocity_y=0.1,
        hip_y_avg=0.5,
        elbow_angle=90.0,
        index_align_angle=170.0,
        index_velocity_y=0.1,
    )

    for _ in range(5):
        det.update(index_release)
    assert det.phase == "release"


def test_unreliable_index_does_not_copy_wrist_velocity():
    current = {
        "right_wrist": {
            "position": [0.0, 0.6, 0.0],
            "is_reliable": True,
        },
        "right_index": {
            "position": [0.0, 0.7, 0.0],
            "is_reliable": False,
        },
    }
    previous = {
        "right_wrist": {
            "position": [0.0, 0.5, 0.0],
            "is_reliable": True,
        },
        "right_index": {
            "position": [0.0, 0.6, 0.0],
            "is_reliable": True,
        },
    }

    features = extract_features(
        current,
        angles={},
        shooting_side="right",
        prev_world=previous,
        dt_s=0.1,
    )
    assert features.wrist_velocity_y > 0
    assert features.index_velocity_y == 0.0


def test_diagnostic_foot_motion_does_not_change_phase_speed():
    current = {
        "right_wrist": {
            "position": [0.0, 0.5, 0.0],
            "is_reliable": True,
        },
        "right_foot_index": {
            "position": [10.0, 10.0, 10.0],
            "is_reliable": True,
        },
    }
    previous = {
        "right_wrist": {
            "position": [0.0, 0.5, 0.0],
            "is_reliable": True,
        },
        "right_foot_index": {
            "position": [0.0, 0.0, 0.0],
            "is_reliable": True,
        },
    }

    features = extract_features(
        current,
        angles={},
        shooting_side="right",
        prev_world=previous,
        dt_s=0.1,
    )
    assert features.total_velocity == 0.0


if __name__ == "__main__":
    test_phase_transitions()
    test_biomechanics_knee_rule()
    test_phase_order_complete()
    test_ready_stance_to_loading_on_wrist_rise()
    test_index_tip_can_confirm_release()
    test_unreliable_index_does_not_copy_wrist_velocity()
    test_diagnostic_foot_motion_does_not_change_phase_speed()
    print("All phase/rule tests passed.")
