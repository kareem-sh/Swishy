import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import (
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)
from utils.performance import get_pose_model_path, resolve_pose_delegate, warn_if_gpu_unavailable


class PoseDetector:
    """MediaPipe Pose Landmarker wrapper for image, video, and live stream."""

    def __init__(self, running_mode, result_callback=None):
        warn_if_gpu_unavailable()
        model_path = get_pose_model_path()
        delegate = resolve_pose_delegate()

        try:
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=delegate,
            )
            self.landmarker = self._create_landmarker(base_options, running_mode, result_callback)
            self._delegate = "gpu" if delegate == python.BaseOptions.Delegate.GPU else "cpu"
        except Exception as exc:
            if delegate != python.BaseOptions.Delegate.CPU:
                print(f"GPU init failed ({exc}); falling back to CPU.")
                base_options = python.BaseOptions(
                    model_asset_path=model_path,
                    delegate=python.BaseOptions.Delegate.CPU,
                )
                self.landmarker = self._create_landmarker(base_options, running_mode, result_callback)
                self._delegate = "cpu"
            else:
                raise

    @staticmethod
    def _create_landmarker(base_options, running_mode, result_callback):
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            result_callback=result_callback,
            min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    @property
    def delegate(self) -> str:
        return self._delegate

    def detect_image(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    def detect_video_frame(self, rgb_image, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def detect_async(self, rgb_image, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.landmarker.detect_async(mp_image, timestamp_ms)
