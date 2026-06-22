"""
One Euro Filter — adaptive low-pass filter for real-time pose smoothing.

Reference: Casiez, Roussel, Vogel (2012) "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems"
"""

import math
from typing import Dict, Optional

import numpy as np

from pose.landmarks import BASKETBALL_LANDMARKS


class OneEuroFilter:
    """
    Adaptive low-pass filter for a single scalar signal.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing when still.
        beta: Speed coefficient. Higher = less lag during fast motion.
        d_cutoff: Cutoff for derivative smoothing.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: Optional[float] = None
        self._t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        dx = (x - self._x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None


class LandmarkFilterBank:
    """One Euro filter per landmark axis in world space."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ):
        self._filters: Dict[str, Dict[str, OneEuroFilter]] = {}
        self._params = (min_cutoff, beta, d_cutoff)

        for name in BASKETBALL_LANDMARKS:
            self._filters[name] = {
                axis: OneEuroFilter(min_cutoff, beta, d_cutoff)
                for axis in ("x", "y", "z")
            }

    def filter_landmarks(self, world_landmarks: Dict[str, dict], timestamp_s: float) -> Dict[str, dict]:
        filtered = {}

        for name, landmark in world_landmarks.items():
            pos = landmark["position"]
            filters = self._filters.get(name)
            if filters is None:
                filtered[name] = landmark
                continue

            filtered_pos = np.array([
                filters["x"].filter(float(pos[0]), timestamp_s),
                filters["y"].filter(float(pos[1]), timestamp_s),
                filters["z"].filter(float(pos[2]), timestamp_s),
            ], dtype=np.float64)

            filtered[name] = {**landmark, "position": filtered_pos}

        return filtered

    def reset(self):
        for axis_filters in self._filters.values():
            for f in axis_filters.values():
                f.reset()
