"""
3D vector mathematics for biomechanical angle computation.

All functions operate on numpy arrays in R^3. Angles are returned in degrees.
"""

import numpy as np

_EPSILON = 1e-8


def segment_vector(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Vector from start point to end point."""
    return np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)


def normalize(vector: np.ndarray) -> np.ndarray:
    """Return unit vector; zero vector stays zero."""
    vector = np.asarray(vector, dtype=np.float64)
    magnitude = np.linalg.norm(vector)
    if magnitude < _EPSILON:
        return vector
    return vector / magnitude


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint between two 3D points."""
    return (np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64)) / 2.0


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Interior angle (0-180 degrees) between two 3D vectors.

    Uses the dot product formula:
        cos(theta) = (v1 . v2) / (|v1| * |v2|)
        theta = arccos(cos(theta))

  This is rotation-invariant: the angle does not change when the
  coordinate frame rotates, unlike 2D arctan2 on image (x, y).
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < _EPSILON or norm2 < _EPSILON:
        return float("nan")

    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def angle_from_vertical(segment: np.ndarray) -> float:
    """
    Angle between a body segment and the world-up axis (negative Y in MediaPipe).

    MediaPipe world coordinates: +Y is up, +X is right, +Z is toward the camera.
    """
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return angle_between_vectors(segment, up)
