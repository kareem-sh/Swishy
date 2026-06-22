"""
Confidence-aware landmark processing for occlusion handling.
"""

from typing import Dict, Optional


class VisibilityGate:
    """
    Gate landmark and angle computations based on MediaPipe visibility scores.

    When a landmark is briefly occluded, the last reliable position can be held
    for a short window to avoid angle flicker.
    """

    def __init__(
        self,
        visibility_threshold: float = 0.6,
        hold_frames: int = 5,
    ):
        self.visibility_threshold = visibility_threshold
        self.hold_frames = hold_frames
        self._hold_counters: Dict[str, int] = {}
        self._last_reliable: Dict[str, dict] = {}

    def is_landmark_reliable(self, landmark: Optional[dict]) -> bool:
        if landmark is None:
            return False
        return landmark.get("is_reliable", False) and landmark.get("visibility", 0.0) >= self.visibility_threshold

    def apply(self, world_landmarks: Dict[str, dict]) -> Dict[str, dict]:
        """Mark each landmark as reliable/unreliable; apply temporal hold."""
        gated = {}

        for name, landmark in world_landmarks.items():
            visibility = landmark.get("visibility", 0.0)
            reliable_now = visibility >= self.visibility_threshold

            if reliable_now:
                self._hold_counters[name] = 0
                self._last_reliable[name] = {
                    "position": landmark["position"].copy(),
                    "visibility": visibility,
                }
                gated[name] = {
                    **landmark,
                    "is_reliable": True,
                    "is_stable": True,
                }
                continue

            hold_count = self._hold_counters.get(name, 0) + 1
            self._hold_counters[name] = hold_count

            if hold_count <= self.hold_frames and name in self._last_reliable:
                last = self._last_reliable[name]
                decay = 1.0 - (hold_count / (self.hold_frames + 1))
                gated[name] = {
                    **landmark,
                    "position": last["position"].copy(),
                    "visibility": last["visibility"] * decay,
                    "is_reliable": True,
                    "is_stable": False,
                }
            else:
                gated[name] = {
                    **landmark,
                    "is_reliable": False,
                    "is_stable": False,
                }

        return gated

    def reset(self):
        self._hold_counters.clear()
        self._last_reliable.clear()
