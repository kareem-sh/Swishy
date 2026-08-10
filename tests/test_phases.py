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
    """A dip with the ball in the carry position enters `loading`.

    The dip is expressed as knee flexion, not as hip velocity. World landmarks
    are hip-centred -- the hip midpoint IS the origin -- so `hip_velocity_y` is
    identically zero on every real frame, and the transition that this test
    used to drive with `hip_velocity_y=-0.05` could never fire outside a
    synthetic fixture.
    """
    det = ShotPhaseDetector()
    assert det.phase == "ready_stance"

    # hip_y_avg defaults to 0.6, so the ball has to be carried at or above
    # that to be in a position it could be shot from. wrist_y=0.4 put the
    # hands two-tenths BELOW the hips, which is a player standing at rest.
    loading = _make_features(knee_angle_delta=-3.0, wrist_y=0.7, shoulder_y=0.8)
    for _ in range(5):
        det.update(loading)
    assert det.phase == "loading"


def test_hip_velocity_alone_cannot_start_a_shot():
    """Guards the bug above: hip velocity is a dead signal, not a weak one.

    If this ever passes, someone has reintroduced a condition that reads
    whole-body motion out of hip-centred coordinates, where it does not exist.
    """
    det = ShotPhaseDetector()
    hip_only = _make_features(hip_velocity_y=-0.05, knee_angle_delta=0.0,
                              wrist_y=0.7, shoulder_y=0.8)
    for _ in range(8):
        det.update(hip_only)
    assert det.phase == "ready_stance"


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


def test_ready_stance_to_ball_lift_on_wrist_rise():
    """A rising wrist from the carry position starts the ball lift.

    Two things changed here. The shoulder now sits ABOVE the hips
    (`shoulder_y=0.4`, not `-0.4`): after the Y-axis correction, up is
    positive, so the old fixture described an anatomically impossible body and
    the carry window could never contain the wrist.

    And a rising wrist now goes straight to `ball_lift` rather than `loading`.
    A wrist travelling upward IS the lift; routing it through `loading` first
    described a dip that is not happening.
    """
    det = ShotPhaseDetector()
    still = _make_features(total_velocity=0.05, wrist_y=0.02, hip_y_avg=0.0,
                           shoulder_y=0.4)
    for _ in range(3):
        det.update(still)

    rising = _make_features(
        total_velocity=0.08,
        wrist_y=0.03,
        wrist_velocity_y=0.05,
        hip_y_avg=0.0,
        shoulder_y=0.4,
    )
    phases = []
    for _ in range(5):
        det.update(rising)
        phases.append(det.phase)
    assert "ball_lift" in phases


def test_wrist_outside_carry_window_does_not_start_a_shot():
    """A hand hanging far below the hips is not a shot about to happen."""
    det = ShotPhaseDetector()
    hanging = _make_features(wrist_y=-0.60, hip_y_avg=0.0, shoulder_y=0.4,
                             wrist_velocity_y=0.05, knee_angle_delta=-3.0)
    for _ in range(8):
        det.update(hanging)
    assert det.phase == "ready_stance"


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
