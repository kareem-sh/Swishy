"""Resolve pose model path and MediaPipe delegate from config."""

import platform
import sys
from pathlib import Path

from mediapipe.tasks import python

from utils.config_loader import load_yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

POSE_MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

POSE_MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def get_pose_model_path() -> str:
    cfg = load_yaml("performance.yaml")
    key = str(cfg.get("pose_model", "full")).lower()
    filename = POSE_MODEL_FILES.get(key, POSE_MODEL_FILES["full"])
    return str(_PROJECT_ROOT / "models" / filename)


def resolve_pose_delegate() -> python.BaseOptions.Delegate:
    cfg = load_yaml("performance.yaml")
    choice = str(cfg.get("pose_delegate", "auto")).lower()

    if choice == "cpu":
        return python.BaseOptions.Delegate.CPU
    if choice == "gpu":
        return python.BaseOptions.Delegate.GPU

    # auto
    if platform.system() == "Windows":
        return python.BaseOptions.Delegate.CPU
    return python.BaseOptions.Delegate.GPU


def warn_if_gpu_unavailable():
    cfg = load_yaml("performance.yaml")
    choice = str(cfg.get("pose_delegate", "auto")).lower()
    if choice in ("gpu",) and platform.system() == "Windows":
        print(
            "Note: MediaPipe GPU is not supported in Python on Windows. Using CPU.\n"
            "  For faster processing, set pose_model: lite in config/performance.yaml",
            file=sys.stderr,
        )
