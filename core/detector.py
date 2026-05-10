import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import (
    MODEL_PATH,
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE
)


class PoseDetector:

    def __init__(
        self,
        running_mode,
        result_callback=None
    ):

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.PoseLandmarkerOptions(

            base_options=base_options,

            running_mode=running_mode,

            result_callback=result_callback,

            min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,

            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,

            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,

            num_poses=1,

            output_segmentation_masks=False
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

    # IMAGE MODE
    def detect_image(self, rgb_image):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect(
            mp_image
        )

        return result

    # VIDEO MODE
    def detect_video_frame(
        self,
        rgb_image,
        timestamp_ms
    ):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    # LIVE STREAM MODE
    def detect_async(
        self,
        rgb_image,
        timestamp_ms
    ):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        self.landmarker.detect_async(
            mp_image,
            timestamp_ms
        )
