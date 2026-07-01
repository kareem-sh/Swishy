import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry.vectors import angle_between_vectors, segment_vector
from angles.calculator import AngleCalculator
from pose.visibility import VisibilityGate


def test_right_angle():
    """Three points forming a 90-degree angle in 3D."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    v1 = segment_vector(b, a)
    v2 = segment_vector(b, c)
    angle = angle_between_vectors(v1, v2)
    assert abs(angle - 90.0) < 0.01


def test_straight_line():
    """Collinear points should give ~180 degrees."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])
    v1 = segment_vector(b, a)
    v2 = segment_vector(b, c)
    angle = angle_between_vectors(v1, v2)
    assert abs(angle - 180.0) < 0.01


def test_rotation_invariance():
    """Angle should not change when coordinate frame rotates."""
    a = np.array([1.0, 0.5, 0.3])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.2, 1.0, 0.1])
    v1 = segment_vector(b, a)
    v2 = segment_vector(b, c)
    angle_before = angle_between_vectors(v1, v2)

    # Rotate all points around Y axis by 45 degrees
    theta = math.radians(45)
    rot = np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)],
    ])
    a_r = rot @ a
    b_r = rot @ b
    c_r = rot @ c
    v1_r = segment_vector(b_r, a_r)
    v2_r = segment_vector(b_r, c_r)
    angle_after = angle_between_vectors(v1_r, v2_r)

    assert abs(angle_before - angle_after) < 0.01


def test_visibility_gates_invalid_angle():
    gate = VisibilityGate(visibility_threshold=0.6, presence_threshold=0.5)
    calc = AngleCalculator(gate)

    world = {
        "right_shoulder": {"position": np.array([0.1, 0.5, 0.0]), "visibility": 0.9, "presence": 0.9, "is_reliable": True, "is_stable": True},
        "right_elbow": {"position": np.array([0.2, 0.4, 0.0]), "visibility": 0.3, "presence": 0.9, "is_reliable": False, "is_stable": False},
        "right_wrist": {"position": np.array([0.3, 0.3, 0.0]), "visibility": 0.9, "presence": 0.9, "is_reliable": True, "is_stable": True},
    }
    world = gate.apply(world)

    from angles.joint_chains import JOINT_CHAINS
    result = calc.compute_joint_angle(world, JOINT_CHAINS["right_elbow"], "right_elbow")
    assert not result.is_valid


def test_presence_gates_high_visibility_low_presence():
    gate = VisibilityGate(visibility_threshold=0.6, presence_threshold=0.5)
    world = {
        "right_knee": {
            "position": np.array([0.0, 0.0, 0.0]),
            "visibility": 0.95,
            "presence": 0.2,
        },
    }
    gated = gate.apply(world)
    assert gated["right_knee"]["is_reliable"] is False
    assert gated["right_knee"]["confidence"] < 0.5


if __name__ == "__main__":
    test_right_angle()
    test_straight_line()
    test_rotation_invariance()
    test_visibility_gates_invalid_angle()
    test_presence_gates_high_visibility_low_presence()
    print("All tests passed.")
