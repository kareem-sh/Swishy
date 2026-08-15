"""
3D joint angle calculator using world-space landmarks.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from angles.joint_chains import JOINT_CHAINS, JointChain
from geometry.vectors import angle_from_vertical, angle_between_vectors, midpoint, segment_vector
from pose.visibility import VisibilityGate


_SIDE_KEYS = ("elbow", "knee", "ankle_flexion", "hip", "shoulder", "index_align")

# Angles of the LEGS, which belong to the player rather than to the shooting
# hand. See compute_all for why only these may be measured on the other side.
_LEG_KEYS = frozenset({"knee", "hip", "ankle_flexion"})


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

        other_side = "left" if shooting_side == "right" else "right"

        for key in _SIDE_KEYS:
            chain_name = f"{shooting_side}_{key}"
            result = self.compute_joint_angle(
                world_landmarks, JOINT_CHAINS[chain_name], chain_name
            )

            # LEG angles fall back to the other leg. Arm angles never do.
            #
            # A shooter loads BOTH legs, so which knee we measure is a question
            # about visibility, not about handedness -- while the elbow, the
            # shoulder and the finger line belong to the shooting arm and
            # nothing else, so substituting the other arm would report a
            # different movement under the same name.
            #
            # Binding the legs to the shooting side threw away perfectly good
            # data. Measured on salah_video shots 3 and 4: the shooting side was
            # read as `left`, so the LEFT knee was measured at visibility 0.48
            # and rejected on all 45 frames -- while the RIGHT knee sat at 0.95
            # and was reliable on all 45. From this camera the far leg is
            # occluded by the near one, so the far knee is unusable whichever
            # hand shoots. Both shots lost their knee AND hip rules, scored on 4
            # rules instead of 9, and came out at 20 and 35 -- for a load the
            # player had performed and the camera had recorded.
            if key in _LEG_KEYS and not result.is_valid:
                fallback_name = f"{other_side}_{key}"
                fallback = self.compute_joint_angle(
                    world_landmarks, JOINT_CHAINS[fallback_name], chain_name
                )
                if fallback.is_valid:
                    result = fallback

            results[chain_name] = result

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

        # Past 90 deg the shoulders are BELOW the hips. Nobody shoots a
        # basketball inverted, so this is not a posture -- it is a pose failure,
        # usually the trunk vector resolved backwards.
        #
        # It has to be caught here rather than by the rule band. `trunk_posture`
        # allows 0-25, so a reading of 159.8 is simply "outside the band" and
        # scores exactly like a real 30 deg lean: the player is told to keep
        # their chest toward the rim, from a frame in which the detector had
        # them upside down. Measured across 53 clips this reached the score 5
        # times on 2 clips, at 97.8 to 159.8 deg.
        #
        # Invalid, not clamped. We know the reading is wrong; we do not know
        # what the trunk was actually doing, and substituting a plausible
        # number would manufacture a measurement.
        if degrees is not None and degrees > 90.0:
            return AngleResult("trunk", None, False, False)

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
