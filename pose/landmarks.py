"""
Landmark extraction from MediaPipe Pose Landmarker results.

Provides both image-space landmarks (for drawing) and world-space landmarks
(for 3D angle computation).
"""

from typing import Dict, Optional

import numpy as np

POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Subset used for basketball analysis and filtering
BASKETBALL_LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def extract_image_landmarks(detection_result, width: int, height: int) -> Optional[Dict[str, dict]]:
    """Extract image-space landmarks with pixel and normalized coordinates."""
    if not detection_result.pose_landmarks:
        return None

    pose = detection_result.pose_landmarks[0]
    data = {}

    for name, idx in BASKETBALL_LANDMARKS.items():
        landmark = pose[idx]
        data[name] = {
            "x_norm": landmark.x,
            "y_norm": landmark.y,
            "z": landmark.z,
            "x": int(landmark.x * width),
            "y": int(landmark.y * height),
            "visibility": landmark.visibility,
            "presence": landmark.presence,
        }

    return data


def extract_world_landmarks(detection_result) -> Optional[Dict[str, dict]]:
    """
    Extract world-space landmarks in meters (MediaPipe body-centric frame).

    Origin: midpoint between hips.
    +Y: up, +X: subject's right, +Z: toward camera.
    """
    if not detection_result.pose_world_landmarks:
        return None

    pose = detection_result.pose_world_landmarks[0]
    data = {}

    for name, idx in BASKETBALL_LANDMARKS.items():
        landmark = pose[idx]
        data[name] = {
            "position": np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64),
            "visibility": landmark.visibility,
            "presence": landmark.presence,
            "is_stable": True,
            "is_reliable": True,
        }

    return data


def extract_all_landmarks(detection_result, width: int, height: int) -> Optional[dict]:
    """Return both image and world landmark dictionaries."""
    image = extract_image_landmarks(detection_result, width, height)
    world = extract_world_landmarks(detection_result)

    if image is None or world is None:
        return None

    return {"image": image, "world": world}
