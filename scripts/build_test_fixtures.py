"""
Capture real MediaPipe world landmarks from the sample videos as test fixtures.

Why this exists
---------------
The pose test-suite used to assert against hand-typed landmark values. Those
fixtures encoded a +Y-up world while MediaPipe actually emitted +Y-down, so the
suite stayed green while the product was measurably broken (trunk rule 0% pass
over 1359 frames). Hand-written fixtures test our mental model, not the model.

These fixtures are recorded from the real model so the tests describe reality.

Regenerate with:
    python scripts/build_test_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mediapipe.tasks.python import vision  # noqa: E402

from pose.detector import PoseDetector  # noqa: E402
from pose.landmarks import extract_all_landmarks  # noqa: E402

FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "real_world_landmarks.json"

# Frames chosen to cover a standing pose plus a shot in flight. Indices were
# picked by inspection of the phase timeline, not tuned to make tests pass.
SAMPLES = [
    ("video_01_free_throw", 10, "standing_ready"),
    ("video_01_free_throw", 60, "mid_shot"),
    ("video_07_side_jump_shot", 40, "standing_ready"),
    ("video_07_side_jump_shot", 300, "mid_shot"),
]


def capture(video_stem: str, frame_index: int, label: str) -> dict | None:
    path = PROJECT_ROOT / "assets" / "videos" / f"{video_stem}.mp4"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  !! cannot open {path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"  !! cannot read frame {frame_index} of {video_stem}")
        return None

    detector = PoseDetector(vision.RunningMode.IMAGE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_image(rgb)
    landmarks = extract_all_landmarks(result, frame.shape[1], frame.shape[0])
    if landmarks is None:
        print(f"  !! no pose detected in {video_stem}:{frame_index}")
        return None

    world = {
        name: {
            "position": [float(v) for v in lm["position"]],
            "visibility": float(lm["visibility"]),
            "presence": float(lm["presence"]),
        }
        for name, lm in landmarks["world"].items()
    }

    return {
        "video": video_stem,
        "frame_index": frame_index,
        "label": label,
        "fps": round(float(fps), 3),
        "world": world,
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for stem, idx, label in SAMPLES:
        print(f"capturing {stem}:{idx} ({label})")
        sample = capture(stem, idx, label)
        if sample is not None:
            samples.append(sample)

    if not samples:
        print("No samples captured — fixtures NOT written.")
        return 1

    payload = {
        "_comment": (
            "Real MediaPipe world landmarks in Swichy canonical coordinates "
            "(+Y UP), recorded by scripts/build_test_fixtures.py. Do not "
            "hand-edit: regenerate instead."
        ),
        "convention": "swichy_canonical_y_up",
        "samples": samples,
    }
    FIXTURE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {len(samples)} samples -> {FIXTURE_PATH.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
