"""Monotonic timestamp helpers for MediaPipe VIDEO and LIVE_STREAM modes."""


def frame_timestamp_ms(frame_index: int, fps: float) -> int:
    """
    Compute a stable, monotonically increasing timestamp in milliseconds.

    MediaPipe VIDEO/LIVE_STREAM modes require timestamps that increase with
    each frame. Using time.time() causes tracking instability.
    """
    if fps <= 0:
        fps = 30.0
    return int(frame_index * 1000.0 / fps)
