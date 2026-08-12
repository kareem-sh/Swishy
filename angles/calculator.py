"""
3D joint angle calculator using world-space landmarks.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from angles.joint_chains import JOINT_CHAINS, JointChain
from geometry.vectors import angle_from_vertical, angle_between_vectors, midpoint, segment_vector
from pose.visibility import VisibilityGate


@dataclass
class AngleResult:
    name: str
    degrees: Optional[float]
    is_valid: bool
    is_stable: bool


class AngleCalculator:
    """Compute joint angles from filtered world landmarks."""

    def __init__(self, visibility_gate: VisibilityGate):
        self._visibility = visibility_gate

    def compute_joint_angle(
        self,
        world_landmarks: Dict[str, dict],
        chain: JointChain,
        chain_name: str,
    ) -> AngleResult:
        resolved = self._resolve_landmarks(world_landmarks, chain)
        if resolved is None:
            return AngleResult(
                name=chain_name,
                degrees=None,
                is_valid=False,
                is_stable=False,
            )

        a, b, c, stable = resolved
        v1 = segment_vector(b, a)
        v2 = segment_vector(b, c)
        degrees = angle_between_vectors(v1, v2)

        if np.isnan(degrees):
            return AngleResult(
                name=chain_name,
                degrees=None,
                is_valid=False,
                is_stable=False,
            )

        return AngleResult(
            name=chain_name,
            degrees=degrees,
            is_valid=True,
            is_stable=stable,
        )

    def compute_all(
        self,
        world_landmarks: Dict[str, dict],
        shooting_side: str = "right",
    ) -> Dict[str, AngleResult]:
        """ 
        Compute all joint angles for the given shooting side plus trunk.

        shooting_side: 'left' or 'right' — determines which limb chain to prioritize.
        """
        results: Dict[str, AngleResult] = {}

        side_keys = (
            "elbow",
            "knee",
            "ankle_flexion",
            "hip",
            "shoulder",
            "index_align",
        )
        for key in side_keys:
            chain_name = f"{shooting_side}_{key}"
            results[chain_name] = self.compute_joint_angle(
                world_landmarks,
                JOINT_CHAINS[chain_name],
                chain_name,
            )

        results["trunk"] = self._compute_trunk_angle(world_landmarks)
        return results

    def _compute_trunk_angle(self, world_landmarks: Dict[str, dict]) -> AngleResult:
        """Trunk lean: angle between hip-mid -> shoulder-mid segment and vertical."""
        for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
            if not self._visibility.is_landmark_reliable(world_landmarks.get(name)):
                return AngleResult("trunk", None, False, False)

        left_hip = world_landmarks["left_hip"]["position"]
        right_hip = world_landmarks["right_hip"]["position"]
        left_shoulder = world_landmarks["left_shoulder"]["position"]
        right_shoulder = world_landmarks["right_shoulder"]["position"]

        hip_mid = midpoint(left_hip, right_hip)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        trunk_vector = segment_vector(hip_mid, shoulder_mid)

        degrees = angle_from_vertical(trunk_vector)
        stable = all(
            world_landmarks[n].get("is_stable", True)
            for n in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
        )
        return AngleResult("trunk", degrees, True, stable)

    def _resolve_landmarks(
        self,
        world_landmarks: Dict[str, dict],
        chain: JointChain,
    ) -> Optional[tuple]:
        names = chain.landmark_names
        points = []
        all_stable = True

        for name in names:
            landmark = world_landmarks.get(name)
            if landmark is None or not self._visibility.is_landmark_reliable(landmark):
                return None
            points.append(landmark["position"])
            if not landmark.get("is_stable", True):
                all_stable = False

        return points[0], points[1], points[2], all_stable
