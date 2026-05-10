# core/landmarks.py


POSE_LANDMARKS = {

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
    "right_ankle": 28
}


def extract_landmarks(
    detection_result,
    width,
    height
):
    """
    Extract important pose landmarks.

    Returns:
    dictionary containing:
    - pixel coordinates
    - normalized coordinates
    - depth
    - visibility
    """

    # No pose detected
    if not detection_result.pose_landmarks:
        return None

    # First detected pose
    pose = detection_result.pose_landmarks[0]

    data = {}

    for name, idx in POSE_LANDMARKS.items():

        landmark = pose[idx]

        # Normalized coordinates (0 -> 1)
        x_norm = landmark.x
        y_norm = landmark.y

        # Pixel coordinates
        x = int(x_norm * width)
        y = int(y_norm * height)

        # Depth
        z = landmark.z

        # Visibility confidence
        visibility = landmark.visibility

        data[name] = {

            # Pixel coordinates
            "x": x,
            "y": y,

            # Normalized coordinates
            "x_norm": x_norm,
            "y_norm": y_norm,

            # Depth
            "z": z,

            # Confidence
            "visibility": visibility
        }

    return data
