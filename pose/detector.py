import logging
import sys

import utils.quiet  # noqa: F401  — must precede mediapipe; silences native logs

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import (
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
)
from utils.config_loader import load_yaml

# Which delegate was selected is diagnostic, not something the player needs to
# read. It goes to the logger so `-v` can surface it without it living in the
# coaching output.
log = logging.getLogger(__name__)


class PoseDetector:
    """MediaPipe Pose Landmarker wrapper for image, video, and live stream."""

    def __init__(self, running_mode, result_callback=None):
        pose_cfg = load_yaml("pose.yaml")
        requested_device = str(pose_cfg.get("device", "auto")).strip().lower()
        if requested_device not in {"auto", "cpu", "gpu"}:
            log.warning(
                "unknown MediaPipe pose device %r; using auto", requested_device
            )
            requested_device = "auto"

        self.device = "cpu"
        should_try_gpu = requested_device == "gpu" or (
            requested_device == "auto" and sys.platform.startswith("linux")
        )
        if should_try_gpu:
            try:
                self.landmarker = self._create_landmarker(
                    running_mode,
                    result_callback,
                    python.BaseOptions.Delegate.GPU,
                )
                self.device = "gpu"
                log.info("MediaPipe pose device: GPU")
                return
            except (RuntimeError, ValueError, NotImplementedError) as exc:
                reason = str(exc).splitlines()[0] or type(exc).__name__
                log.info(
                    "MediaPipe pose GPU unavailable (%s); falling back to CPU", reason
                )

        self.landmarker = self._create_landmarker(
            running_mode,
            result_callback,
            python.BaseOptions.Delegate.CPU,
        )
        log.info("MediaPipe pose device: CPU")

    @staticmethod
    def _create_landmarker(running_mode, result_callback, delegate):
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=delegate,
        )
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

    def detect_image(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect(mp_image)

    def detect_video_frame(self, rgb_image, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def detect_async(self, rgb_image, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.landmarker.detect_async(mp_image, timestamp_ms)
