"""Deprecated: use pose.landmarks instead."""
from pose.landmarks import BASKETBALL_LANDMARKS as POSE_LANDMARKS
from pose.landmarks import extract_image_landmarks as extract_landmarks

__all__ = ["POSE_LANDMARKS", "extract_landmarks"]
