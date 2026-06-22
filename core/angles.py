"""
Deprecated: 2D angle calculation.

Use angles.calculator with 3D world landmarks instead.
Kept only for reference comparison.
"""

import numpy as np


def calculate_angle(a, b, c):
    """Legacy 2D image-plane angle. Do not use for production analysis."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0])
        - np.arctan2(a[1] - b[1], a[0] - b[0])
    )
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle
