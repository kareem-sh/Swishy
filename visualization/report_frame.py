"""Enhanced frame annotation for saved report images."""

import cv2
import numpy as np

from feedback.report_models import KeyFrame


def annotate_key_frame(bgr_image: np.ndarray, key_frame: KeyFrame) -> np.ndarray:
    """Add report callout box explaining what to improve on this frame."""
    img = bgr_image.copy()
    h, w, _ = img.shape

    header = f"{key_frame.phase_label} @ {key_frame.timestamp_label}"
    cv2.rectangle(img, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(img, header, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if key_frame.issues:
        box_h = min(30 + 28 * len(key_frame.issues), h // 2)
        cv2.rectangle(img, (0, h - box_h), (w, h), (0, 0, 0), -1)
        cv2.putText(img, "IMPROVE HERE:", (10, h - box_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y = h - box_h + 48
        for issue in key_frame.issues[:4]:
            text = issue if len(issue) <= 55 else issue[:52] + "..."
            cv2.putText(img, f"! {text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y += 24

    if key_frame.angles_summary:
        cv2.putText(
            img, key_frame.angles_summary, (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
        )

    return img
