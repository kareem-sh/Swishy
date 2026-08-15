"""Request-scoped physical inputs for backend video analysis."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from pipeline import ShotAnalysisPipeline
from player.profile import build_player_profile


def test_pipeline_request_inputs_override_yaml_without_mutating_it() -> None:
    pipeline = ShotAnalysisPipeline(
        enable_ball=False,
        player=build_player_profile(height_cm=188, warn=None),
        rim_height_m=3.10,
        shot_xy_m=(2.0, 7.0),
    )

    assert math.isclose(pipeline.rim_height_m, 3.10)
    assert pipeline.court_shot_xy_m == (2.0, 7.0)
    assert math.isclose(
        pipeline._court_shot_distance_m,
        math.hypot(2.0, 7.0 - 1.575),
    )
    assert math.isclose(pipeline._ideal_trajectory.rim_height_m, 3.10)


def test_main_analyze_video_returns_json_compatible_payload() -> None:
    expected = {
        "video": "upload.mov",
        "analysis_inputs": {
            "height_cm": 188.0,
            "rim_height_m": 3.10,
            "shot_xy_m": [2.0, 7.0],
        },
        "shots": [],
    }
    fake_run = SimpleNamespace(to_payload=lambda: expected)

    with patch.object(main, "_analyze_video_run", return_value=fake_run) as run:
        payload = main.analyze_video(
            "upload.mov",
            height_cm=188,
            rim_height_m=3.10,
            shot_xy_m=[2.0, 7.0],
        )

    assert payload is expected
    assert isinstance(payload, dict)
    kwargs = run.call_args.kwargs
    assert kwargs["height_cm"] == 188
    assert kwargs["rim_height_m"] == 3.10
    assert kwargs["shot_xy_m"] == (2.0, 7.0)
    assert kwargs["keep_landmarks"] is False


def test_invalid_explicit_rim_height_is_rejected() -> None:
    try:
        ShotAnalysisPipeline(enable_ball=False, rim_height_m=-1.0)
    except ValueError as exc:
        assert "rim_height_m" in str(exc)
    else:
        raise AssertionError("negative rim height should be rejected")


if __name__ == "__main__":
    test_pipeline_request_inputs_override_yaml_without_mutating_it()
    test_main_analyze_video_returns_json_compatible_payload()
    test_invalid_explicit_rim_height_is_rejected()
    print("All backend input tests passed.")
