"""
Confidence-aware landmark processing for occlusion handling.

Uses both MediaPipe visibility (in-frame, not occluded) and presence
(likelihood the landmark exists in the image) for more reliable gating.
"""

from typing import Dict, Optional


class VisibilityGate:
    """
    Gate landmark and angle computations based on MediaPipe confidence scores.

    Effective confidence = min(visibility, presence) so a landmark must score
    well on both signals before angles trust it.

    When a landmark is briefly occluded, the last reliable position can be held
    for a short window to avoid angle flicker.
    """

    def __init__(
        self,
        visibility_threshold: float = 0.6,
        presence_threshold: float = 0.5,
        hold_frames: int = 5,
        require_presence: bool = True,
    ):
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold
        self.confidence_threshold = visibility_threshold
        self.hold_frames = hold_frames
        self.require_presence = require_presence
        self._hold_counters: Dict[str, int] = {}
        self._last_reliable: Dict[str, dict] = {}

    @staticmethod
    def landmark_confidence(landmark: Optional[dict]) -> float:
        """Conservative score: both visibility and presence must be strong."""
        if landmark is None:
            return 0.0
        visibility = float(landmark.get("visibility", 0.0))
        presence = landmark.get("presence")
        if presence is None:
            return visibility
        return min(visibility, float(presence))

    def is_landmark_reliable(self, landmark: Optional[dict]) -> bool:
        if landmark is None:
            return False
        if not landmark.get("is_reliable", False):
            return False
        confidence = landmark.get("confidence", self.landmark_confidence(landmark))
        return confidence >= self.confidence_threshold

    def _passes_thresholds(self, landmark: dict) -> bool:
        visibility = float(landmark.get("visibility", 0.0))
        if visibility < self.visibility_threshold:
            return False
        if not self.require_presence:
            return True
        presence = landmark.get("presence")
        if presence is None:
            return visibility >= self.visibility_threshold
        return float(presence) >= self.presence_threshold

    def apply(self, world_landmarks: Dict[str, dict]) -> Dict[str, dict]:
        """Mark each landmark as reliable/unreliable; apply temporal hold."""
        gated = {}

        for name, landmark in world_landmarks.items():
            confidence = self.landmark_confidence(landmark)
            reliable_now = self._passes_thresholds(landmark) and confidence >= self.confidence_threshold

            if reliable_now:
                self._hold_counters[name] = 0
                self._last_reliable[name] = {
                    "position": landmark["position"].copy(),
                    "visibility": float(landmark.get("visibility", 0.0)),
                    "presence": float(landmark.get("presence", landmark.get("visibility", 0.0))),
                    "confidence": confidence,
                }
                gated[name] = {
                    **landmark,
                    "confidence": confidence,
                    "is_reliable": True,
                    "is_stable": True,
                }
                continue

            hold_count = self._hold_counters.get(name, 0) + 1
            self._hold_counters[name] = hold_count

            if hold_count <= self.hold_frames and name in self._last_reliable:
                last = self._last_reliable[name]
                decay = 1.0 - (hold_count / (self.hold_frames + 1))
                held_confidence = last["confidence"] * decay
                gated[name] = {
                    **landmark,
                    "position": last["position"].copy(),
                    "visibility": last["visibility"] * decay,
                    "presence": last.get("presence", last["visibility"]) * decay,
                    "confidence": held_confidence,
                    "is_reliable": held_confidence >= self.confidence_threshold * 0.85,
                    "is_stable": False,
                }
            else:
                gated[name] = {
                    **landmark,
                    "confidence": confidence,
                    "is_reliable": False,
                    "is_stable": False,
                }

        return gated

    def reset(self):
        self._hold_counters.clear()
        self._last_reliable.clear()
