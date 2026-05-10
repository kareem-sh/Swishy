import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate angle between 3 points.

    Parameters:
    a, b, c -> tuples/lists like:
    (x, y)

    Angle is calculated at point b.

    Example:
    shoulder, elbow, wrist
    """

    # Convert to numpy arrays
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # Calculate radians
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0]) -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    # Convert radians to degrees
    angle = np.abs(np.degrees(radians))

    # Keep angle between 0 and 180
    if angle > 180:
        angle = 360 - angle

    return angle
