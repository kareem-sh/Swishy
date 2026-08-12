"""Tests for phase detection and biomechanical rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.engine import BiomechanicsEngine
from angles.calculator import AngleResult
from phase_detection.detector import ShotPhaseDetector
from phase_detection.features import KinematicFeatures, extract_features
from phase_detection.phases import (
    CORE_ANCHOR,
    CORE_REST,
    CORE_STATES,
    PHASE_ORDER,
)


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


def test_knee_dip_alone_does_not_start_a_shot():
    """A dip is not a shot, and no longer opens one.

    This used to enter a `loading` state. That state is gone from the detector
    on purpose: opening on a posture change meant every re-settle and bounce
    on the toes started an attempt, and it forced the shot through four more
    threshold gates before it could reach a release. The dip is still scored --
    the tracker back-fills frames from before the rise, and
    feedback.phase_refiner finds the dip by its knee minimum -- but it decides
    nothing.
    """
    det = ShotPhaseDetector()
    assert det.phase == CORE_REST

    dipping = _make_features(knee_angle_delta=-3.0, wrist_y=0.7, shoulder_y=0.8)
    for _ in range(8):
        det.update(dipping)
    assert det.phase == CORE_REST


def test_hip_velocity_alone_cannot_start_a_shot():
    """Hip velocity is a dead signal, not a weak one.

    If this ever fails, someone has reintroduced a condition that reads
    whole-body motion out of hip-centred coordinates, where it does not exist:
    the hip midpoint IS the origin, so `hip_velocity_y` is identically zero on
    every real frame.
    """
    det = ShotPhaseDetector()
    hip_only = _make_features(hip_velocity_y=-0.05, knee_angle_delta=0.0,
                              wrist_y=0.7, shoulder_y=0.8)
    for _ in range(8):
        det.update(hip_only)
    assert det.phase == CORE_REST


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


def test_detector_stays_small():
    """The state machine must not grow back.

    Every detector state is another place a shot can get stuck and be lost
    outright: with eight of them, a ten-shot practice video yielded 4 shots and
    a side-on jump shot yielded 0 with 7 discarded candidates. Coaching detail
    is added in config/phase_model.yaml under `analysis`, which costs nothing,
    NOT under `detector`, which costs shots.
    """
    assert len(CORE_STATES) <= 4
    assert CORE_ANCHOR in CORE_STATES
    assert CORE_REST == CORE_STATES[0]


def test_phase_order_complete():
    """The coaching vocabulary is independent of the detector's."""
    assert len(PHASE_ORDER) == 7
    assert PHASE_ORDER[0] == "ready_stance"
    assert PHASE_ORDER[-1] == "landing"
    # `knee_flexion` was removed: it carried exactly the same two rules as
    # `loading`, so one dip was reported twice under two headings.
    assert "knee_flexion" not in PHASE_ORDER


def test_rest_to_rise_on_wrist_rise():
    """A rising wrist from the carry position starts the shot.

    The shoulder sits ABOVE the hips (`shoulder_y=0.4`, not `-0.4`): after the
    Y-axis correction, up is positive, so the old fixture described an
    anatomically impossible body and the carry window could never contain the
    wrist.
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
    assert "rise" in phases


def test_wrist_outside_carry_window_does_not_start_a_shot():
    """A hand hanging far below the hips is not a shot about to happen."""
    det = ShotPhaseDetector()
    hanging = _make_features(wrist_y=-0.60, hip_y_avg=0.0, shoulder_y=0.4,
                             wrist_velocity_y=0.05, knee_angle_delta=-3.0)
    for _ in range(8):
        det.update(hanging)
    assert det.phase == CORE_REST


def test_index_tip_can_confirm_release():
    """A finger driving through the ball confirms the release.

    The hand is carried up first, because a release condition is only
    considered once the ball has actually been on its way up -- see
    ShotPhaseDetector._ball_went_up. Feeding the release frame to a detector
    that never saw a carry tests a situation that cannot occur.
    """
    det = ShotPhaseDetector()
    det.phase = "rise"
    for wrist_y in (0.30, 0.42, 0.54, 0.66):
        det.update(_make_features(
            wrist_y=wrist_y, wrist_velocity_y=0.1, hip_y_avg=0.20,
            shoulder_y=0.55, elbow_angle=90.0, index_align_angle=170.0,
        ))

    index_release = _make_features(
        wrist_y=0.66,
        wrist_velocity_y=0.1,
        hip_y_avg=0.20,
        shoulder_y=0.55,
        elbow_angle=90.0,
        index_align_angle=170.0,
        index_velocity_y=0.1,
    )
    for _ in range(5):
        det.update(index_release)
    assert det.phase == "release"


def test_release_needs_the_ball_to_have_gone_up():
    """An extended arm with no carry is not a release.

    Merging `ball_lift` and `jump` into one `rise` state removed an implicit
    guard -- reaching `jump` used to prove the ball had been carried up. Without
    a replacement, the opening frames of a clip (One Euro filter, ankle
    baseline and every velocity still settling from nothing) produced a release
    inside 0.3 s on three separate fixtures.
    """
    det = ShotPhaseDetector()
    det.phase = "rise"
    # Arm extended, finger aligned, hand well below the shoulder, never moving.
    still = _make_features(
        wrist_y=0.30, wrist_velocity_y=0.0, hip_y_avg=0.20, shoulder_y=0.90,
        elbow_angle=175.0, index_align_angle=175.0,
    )
    for _ in range(10):
        det.update(still)
    assert det.phase == "rise"


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
    import pytest

    raise SystemExit(pytest.main([__file__]))
