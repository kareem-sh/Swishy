import cv2
import numpy as np

from core.angles import calculate_angle


# ==========================================
# MANUAL POSE CONNECTIONS
# ==========================================

POSE_CONNECTIONS = [

    # Face to shoulders
    (0, 11),
    (0, 12),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28)
]


# ==========================================
# DRAW FUNCTION
# ==========================================

def draw_landmarks_on_image(
    rgb_image,
    detection_result,
    landmarks_data=None
):

    annotated_image = np.copy(rgb_image)

    if not detection_result.pose_landmarks:
        return annotated_image

    height, width, _ = annotated_image.shape

    # ==========================================
    # LOOP THROUGH DETECTED POSES
    # ==========================================

    for pose_landmarks in detection_result.pose_landmarks:

        # ==========================================
        # DRAW CONNECTIONS
        # ==========================================

        for connection in POSE_CONNECTIONS:

            start_idx, end_idx = connection

            start_landmark = pose_landmarks[start_idx]
            end_landmark = pose_landmarks[end_idx]

            start_point = (
                int(start_landmark.x * width),
                int(start_landmark.y * height)
            )

            end_point = (
                int(end_landmark.x * width),
                int(end_landmark.y * height)
            )

            cv2.line(
                annotated_image,
                start_point,
                end_point,
                (0, 255, 0),
                2
            )

        # ==========================================
        # DRAW LANDMARK POINTS + IDS
        # ==========================================

        for idx, landmark in enumerate(pose_landmarks):

            pixel_x = int(landmark.x * width)
            pixel_y = int(landmark.y * height)

            # Draw point
            cv2.circle(
                annotated_image,
                (pixel_x, pixel_y),
                5,
                (255, 0, 0),
                -1
            )

            # Draw landmark index
            cv2.putText(
                annotated_image,
                str(idx),
                (pixel_x, pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        # ==========================================
        # ELBOW ANGLE
        # ==========================================

        shoulder = pose_landmarks[12]
        elbow = pose_landmarks[14]
        wrist = pose_landmarks[16]

        shoulder_coords = [shoulder.x, shoulder.y]
        elbow_coords = [elbow.x, elbow.y]
        wrist_coords = [wrist.x, wrist.y]

        elbow_angle = calculate_angle(
            shoulder_coords,
            elbow_coords,
            wrist_coords
        )

        elbow_x = int(elbow.x * width)
        elbow_y = int(elbow.y * height)

        # Coaching feedback
        if elbow_angle < 70:

            elbow_color = (0, 0, 255)
            elbow_text = "Elbow Too Bent"

        else:

            elbow_color = (0, 255, 0)
            elbow_text = "Good Elbow"

        # Draw elbow angle
        cv2.putText(
            annotated_image,
            f"Elbow: {int(elbow_angle)}",
            (elbow_x, elbow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            elbow_color,
            2
        )

        # Draw coaching text
        cv2.putText(
            annotated_image,
            elbow_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            elbow_color,
            2
        )

        # ==========================================
        # KNEE ANGLE
        # ==========================================

        hip = pose_landmarks[24]
        knee = pose_landmarks[26]
        ankle = pose_landmarks[28]

        hip_coords = [hip.x, hip.y]
        knee_coords = [knee.x, knee.y]
        ankle_coords = [ankle.x, ankle.y]

        knee_angle = calculate_angle(
            hip_coords,
            knee_coords,
            ankle_coords
        )

        knee_x = int(knee.x * width)
        knee_y = int(knee.y * height)

        cv2.putText(
            annotated_image,
            f"Knee: {int(knee_angle)}",
            (knee_x, knee_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    return annotated_image
