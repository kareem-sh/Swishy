"""Tests for cached configuration and bounded temporal storage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_loader import clear_config_cache, load_yaml
from utils.frame_buffer import FrameBuffer, FrameSnapshot


def test_cached_config_returns_independent_copies():
    clear_config_cache()
    first = load_yaml("display.yaml")
    second = load_yaml("display.yaml")

    first["video_playback_speed"] = 99
    assert second.get("video_playback_speed") != 99


def test_frame_buffer_is_bounded_and_keeps_latest_frames():
    buffer = FrameBuffer(max_frames=3)
    for index in range(4):
        buffer.push(
            FrameSnapshot(
                timestamp_ms=index * 33,
                angles={},
                shooting_side="right",
            )
        )

    assert len(buffer.frames) == 3
    assert buffer.frames[0].timestamp_ms == 33
    assert buffer.previous is not None
    assert buffer.previous.timestamp_ms == 66
    assert buffer.latest is not None
    assert buffer.latest.timestamp_ms == 99


if __name__ == "__main__":
    test_cached_config_returns_independent_copies()
    test_frame_buffer_is_bounded_and_keeps_latest_frames()
    print("Runtime utility tests passed.")
