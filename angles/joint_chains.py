"""
Named joint angle definitions.

Each joint angle is defined by three landmarks (A, B, C) where B is the vertex.
Vectors are B->A and B->C; the interior angle at B is computed in 3D.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class JointChain:
    """Three landmark names forming a joint angle at the middle landmark."""

    proximal: str
    vertex: str
    distal: str

    @property
    def landmark_names(self) -> Tuple[str, str, str]:
        return (self.proximal, self.vertex, self.distal)


def _side_chains(side: str) -> Dict[str, JointChain]:
    """Build left or right side joint chains. side is 'left' or 'right'."""
    return {
        f"{side}_elbow": JointChain(
            proximal=f"{side}_shoulder",
            vertex=f"{side}_elbow",
            distal=f"{side}_wrist",
        ),
        f"{side}_knee": JointChain(
            proximal=f"{side}_hip",
            vertex=f"{side}_knee",
            distal=f"{side}_ankle",
        ),
        f"{side}_ankle_flexion": JointChain(
            proximal=f"{side}_knee",
            vertex=f"{side}_ankle",
            distal=f"{side}_foot_index",
        ),
        f"{side}_hip": JointChain(
            proximal=f"{side}_shoulder",
            vertex=f"{side}_hip",
            distal=f"{side}_knee",
        ),
        f"{side}_shoulder": JointChain(
            proximal=f"{side}_hip",
            vertex=f"{side}_shoulder",
            distal=f"{side}_elbow",
        ),
        f"{side}_index_align": JointChain(
            proximal=f"{side}_elbow",
            vertex=f"{side}_wrist",
            distal=f"{side}_index",
        ),
    }


JOINT_CHAINS: Dict[str, JointChain] = {
    **_side_chains("left"),
    **_side_chains("right"),
    "trunk": JointChain(
        proximal="mid_hip",
        vertex="mid_shoulder",
        distal="nose",
    ),
}
