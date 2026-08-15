"""Score shot-type detection against clips whose filenames carry the label.

MEASUREMENT ONLY. Reads labels, never writes thresholds.

The label comes from the filename suffix, which is how the dataset was handed
over: `..._set.mp4` is a set shot, `..._jump.mp4` / `_jumpshot` / `_jumpshoot`
is a jump shot, and anything else is unlabelled and reported without a verdict.
Unlabelled clips are still measured, because seeing where they land is how you
notice a threshold sitting in the middle of a cluster.

Usage:
    python scripts/eval_shot_types.py assets/videos/ShootingVideosDataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from pipeline import ShotAnalysisPipeline  # noqa: E402
from player.profile import PlayerProfile  # noqa: E402
from pose.detector import PoseDetector  # noqa: E402
from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from shots.elevation import shooting_event_ms, takeoff_elevation  # noqa: E402

# Longest suffix first, so `_jumpshoot` is not matched by `_jump`.
_JUMP_SUFFIXES = ("jumpshoot", "jumpshot", "jump")
_SET_SUFFIXES = ("set",)


def label_of(name: str) -> Optional[str]:
    stem = Path(name).stem.lower()
    for suffix in _JUMP_SUFFIXES:
        if suffix in stem:
            return "jump"
    for suffix in _SET_SUFFIXES:
        if stem.endswith(f"_{suffix}"):
            return "set"
    return None


def measure(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"clip": path.name, "error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = PoseDetector(vision.RunningMode.VIDEO)
    pipe = ShotAnalysisPipeline(enable_ball=False, player=PlayerProfile())
    pipe.set_fps(fps)

    timestamps: List[int] = []
    ankles: List[Optional[float]] = []
    heights: List[float] = []
    wrists: List[Optional[float]] = []
    posed = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = int(index * 1000.0 / fps)
        result = pipe.process_frame(
            detector.detect_video_frame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts
            ),
            frame.shape[1],
            frame.shape[0],
            ts,
            None,
        )
        f = result.features
        timestamps.append(ts)
        ankles.append(f.ankle_image_y if f else None)
        heights.append(f.body_pixel_height if f else 0.0)
        wrists.append(f.wrist_height_ratio if f else None)
        posed += bool(result.has_pose)
        index += 1
    cap.release()

    event = shooting_event_ms(timestamps, wrists)
    elevation = (
        takeoff_elevation(timestamps, ankles, heights, wrists, event)
        if event is not None
        else None
    )
    return {
        "clip": path.name,
        "label": label_of(path.name),
        "fps": round(fps, 2),
        "frames": index,
        "duration_s": round(index / fps, 2) if fps else None,
        "pose_pct": round(100.0 * posed / max(index, 1), 1),
        "event_s": None if event is None else round(event / 1000.0, 2),
        "elevation": None if elevation is None else round(elevation, 4),
        "predicted": (
            None
            if elevation is None
            else ("jump" if elevation >= JUMP_VERTICAL_DISPLACEMENT_RATIO else "set")
        ),
    }


def report(rows: List[dict]) -> None:
    print()
    print(f"{'clip':<46}{'truth':>6}{'elev':>9}{'pred':>7}{'pose%':>7}{'':>4}")
    print("-" * 82)
    correct = graded = unmeasured = 0
    for r in sorted(rows, key=lambda r: (r.get("label") or "zz", r["clip"])):
        if "error" in r:
            print(f"{r['clip']:<46}{'ERROR':>6}  {r['error']}")
            continue
        elev = "None" if r["elevation"] is None else f"{r['elevation']:.3f}"
        mark = ""
        if r["elevation"] is None:
            unmeasured += 1
            mark = "  --"
        elif r["label"]:
            graded += 1
            good = r["predicted"] == r["label"]
            correct += good
            mark = "  OK" if good else "  X"
        print(
            f"{r['clip']:<46}{str(r['label'] or '-'):>6}{elev:>9}"
            f"{str(r['predicted'] or '-'):>7}{r['pose_pct']:>7}{mark}"
        )

    print("-" * 82)
    if graded:
        print(f"labelled and measured: {correct}/{graded} correct")
    print(f"unmeasurable (no stance observed): {unmeasured}/{len(rows)}")

    def band(kind: str) -> Tuple[Optional[float], Optional[float]]:
        vals = [
            r["elevation"]
            for r in rows
            if r.get("label") == kind and r.get("elevation") is not None
        ]
        return (min(vals), max(vals)) if vals else (None, None)

    set_lo, set_hi = band("set")
    jump_lo, jump_hi = band("jump")
    print(f"threshold in use: {JUMP_VERTICAL_DISPLACEMENT_RATIO}")
    if set_hi is not None:
        print(f"  set  range {set_lo:+.3f} .. {set_hi:+.3f}")
    if jump_lo is not None:
        print(f"  jump range {jump_lo:+.3f} .. {jump_hi:+.3f}")
    if set_hi is not None and jump_lo is not None:
        print(f"  separation gap: {jump_lo - set_hi:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    clips = sorted(Path(args.directory).glob("*.mp4"))
    if not clips:
        print(f"no .mp4 files in {args.directory}")
        return 1

    rows = []
    for i, clip in enumerate(clips, 1):
        print(f"[{i}/{len(clips)}] {clip.name}", flush=True)
        rows.append(measure(clip))

    report(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
