"""
Record a BASELINE of the current pipeline's behaviour on a video, without
changing anything.

Purpose: find out whether the shot-analysis pipeline generalises beyond the
jump-shot / free-throw examples it was built around, BEFORE any threshold is
touched. Reports durations in SECONDS, never as frame percentages, because
some sample material is slow-motion and frame counts are not comparable
across capture rates.

Usage:
    python scripts/generalization_baseline.py <video> [<video> ...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mediapipe.tasks.python import vision  # noqa: E402

from pipeline import ShotAnalysisPipeline  # noqa: E402
from player.profile import PlayerProfile  # noqa: E402
from pose.detector import PoseDetector  # noqa: E402

# Phases that only make sense for a shot involving a vertical two-foot jump.
# Used to detect whether jump-shot assumptions are leaking into other actions.
JUMP_SPECIFIC_PHASES = {"jump"}


def analyse(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = PoseDetector(vision.RunningMode.VIDEO)
    # No height: keeps the baseline independent of any assumed anthropometry.
    pipe = ShotAnalysisPipeline(enable_ball=False, player=PlayerProfile())
    pipe.set_fps(fps)

    transitions = []          # (timestamp_s, phase)
    phase_frames = Counter()
    pose_frames = 0
    shots = []
    warnings = []
    prev_phase = None
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_ms = int(index * 1000.0 / fps)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pipe.process_frame(
            detector.detect_video_frame(rgb, timestamp_ms),
            frame.shape[1],
            frame.shape[0],
            timestamp_ms,
            frame,
        )

        if result.has_pose:
            pose_frames += 1
        phase_frames[result.phase] += 1
        if result.phase != prev_phase:
            transitions.append((round(timestamp_ms / 1000.0, 2), result.phase))
            prev_phase = result.phase
        if result.capture_warning and result.capture_warning not in warnings:
            warnings.append(result.capture_warning)
        if result.shot_summary is not None:
            shots.append(result.shot_summary)
        index += 1

    cap.release()
    trailing = pipe.finalize_session()
    if trailing is not None:
        shots.append(trailing)

    phase_seconds = {p: round(n / fps, 2) for p, n in phase_frames.items()}
    reached = set(phase_frames)

    return {
        "video": video_path.name,
        "fps": round(fps, 2),
        "frames": total_frames,
        "duration_s": round(total_frames / fps, 2),
        "pose_detected_pct": round(100.0 * pose_frames / max(index, 1), 1),
        "detected_shots": len(shots),
        "shot_scores": [s.score for s in shots],
        "shot_phase_sets": [sorted(set(s.phases_seen)) for s in shots],
        "transitions": transitions,
        "phase_seconds": phase_seconds,
        "release_reached": "release" in reached,
        "follow_through_reached": "follow_through" in reached,
        "landing_reached": "landing" in reached,
        "jump_phase_seconds": phase_seconds.get("jump", 0.0),
        "warnings": warnings,
    }


def report(r: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  {r['video']}")
    print("=" * 74)
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return

    print(f"  {r['fps']} fps · {r['frames']} frames · {r['duration_s']} s "
          f"· pose detected on {r['pose_detected_pct']}% of frames")
    print(f"  Detected shots : {r['detected_shots']}  scores={r['shot_scores']}")
    print(f"  release={r['release_reached']}  follow_through={r['follow_through_reached']}"
          f"  landing={r['landing_reached']}")

    print("\n  Phase time (seconds, not frame %):")
    for phase, secs in sorted(r["phase_seconds"].items(), key=lambda kv: -kv[1]):
        share = 100.0 * secs / max(r["duration_s"], 0.01)
        print(f"    {phase:<16} {secs:6.2f}s  ({share:4.1f}% of clip)")

    print(f"\n  Transitions ({len(r['transitions'])}):")
    line = "    " + " -> ".join(f"{p}@{t}s" for t, p in r["transitions"][:18])
    print(line + ("  ..." if len(r["transitions"]) > 18 else ""))

    if r["warnings"]:
        print("\n  Warnings:")
        for w in r["warnings"]:
            print(f"    - {w}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    results = []
    for path in args.videos:
        p = Path(path)
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        r = analyse(p)
        results.append(r)
        report(r)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
