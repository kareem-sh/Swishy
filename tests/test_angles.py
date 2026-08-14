import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry.vectors import angle_between_vectors, segment_vector
from angles.calculator import AngleCalculator
from angles.joint_chains import JOINT_CHAINS
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


def test_shoe_tip_supports_visibility_gated_ankle_flexion():
    gate = VisibilityGate(visibility_threshold=0.6, presence_threshold=0.5)
    calc = AngleCalculator(gate)
    world = {
        "right_knee": {
            "position": np.array([0.0, 1.0, 0.0]),
            "visibility": 0.9,
            "presence": 0.9,
        },
        "right_ankle": {
            "position": np.array([0.0, 0.0, 0.0]),
            "visibility": 0.9,
            "presence": 0.9,
        },
        "right_foot_index": {
            "position": np.array([1.0, 0.0, 0.0]),
            "visibility": 0.9,
            "presence": 0.9,
        },
    }
    gated = gate.apply(world)

    result = calc.compute_joint_angle(
        gated,
        JOINT_CHAINS["right_ankle_flexion"],
        "right_ankle_flexion",
    )
    assert result.is_valid
    assert abs(result.degrees - 90.0) < 0.01

    gated["right_foot_index"]["is_reliable"] = False
    invalid = calc.compute_joint_angle(
        gated,
        JOINT_CHAINS["right_ankle_flexion"],
        "right_ankle_flexion",
    )
    assert not invalid.is_valid


if __name__ == "__main__":
    test_right_angle()
    test_straight_line()
    test_rotation_invariance()
    test_visibility_gates_invalid_angle()
    test_presence_gates_high_visibility_low_presence()
    test_shoe_tip_supports_visibility_gated_ankle_flexion()
    print("All tests passed.")


def test_leg_angles_fall_back_to_the_visible_leg():
    """A knee is a property of the player, not of the shooting hand.

    Regression for salah_video shots 3 and 4. The shooting side was read as
    `left`, so the LEFT knee was measured at visibility 0.48 and rejected on
    all 45 frames, while the RIGHT knee sat at 0.95 and was reliable on all 45
    -- the far leg is occluded by the near one from that camera whichever hand
    shoots. Both shots lost their knee and hip rules, were scored on 4 rules
    instead of 9, and reported 20 and 35 for a load the player had performed
    and the camera had recorded.
    """
    import numpy as np
    from angles.calculator import AngleCalculator
    from pose.visibility import VisibilityGate

    def lm(x, y, z=0.0, vis=1.0):
        return {"position": np.array([x, y, z], float), "visibility": vis,
                "presence": vis, "is_stable": True, "is_reliable": vis >= 0.6,
                "confidence": vis}

    # Left leg unreadable, right leg clean and bent to 90 degrees.
    world = {
        "left_hip": lm(-0.1, 0.0, 0.0, 0.2),
        "left_knee": lm(-0.1, -0.4, 0.0, 0.2),
        "left_ankle": lm(-0.1, -0.8, 0.0, 0.2),
        "left_shoulder": lm(-0.1, 0.5, 0.0, 0.2),
        "right_hip": lm(0.1, 0.0, 0.0),
        "right_knee": lm(0.1, -0.4, 0.0),
        "right_ankle": lm(0.1, -0.4, 0.4),
        "right_shoulder": lm(0.1, 0.5, 0.0),
    }
    calc = AngleCalculator(VisibilityGate())
    angles = calc.compute_all(world, "left")

    knee = angles["left_knee"]
    assert knee.is_valid, "the visible right knee should have been used"
    assert abs(knee.degrees - 90.0) < 1.0, knee.degrees

    # The ARM must NOT fall back: the other arm is a different movement.
    world["left_elbow"] = lm(-0.3, 0.3, 0.0, 0.2)
    world["left_wrist"] = lm(-0.3, 0.6, 0.0, 0.2)
    world["right_elbow"] = lm(0.3, 0.3, 0.0)
    world["right_wrist"] = lm(0.3, 0.6, 0.0)
    assert not calc.compute_all(world, "left")["left_elbow"].is_valid
