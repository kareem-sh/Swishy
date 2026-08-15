"""
Cut video8.mov and video9.mov into single-shot acceptance clips.

WHY
---
Whole-video fixtures answer "did the run go well?" A clip per attempt answers
"which attempt broke, and at which stage?" -- segmentation, classification or
scoring. That is the difference between a number that moves and a defect you
can act on.

GROUND TRUTH
------------
The attempt windows below were NOT produced by Swichy. Each was located with
raw MediaPipe (the frames where the hands rise above the head) and then
confirmed by watching the footage frame by frame:

  video8  10 attempts, fixed wide camera, no cuts. Shots 1-9 keep the feet on
          the floor -- the player rises onto the toes, nothing more. Shot 10
          is a genuine jump, just a low one: the ankles trace a ballistic arc,
          flat at image-y 0.7133, peaking at 0.6749, back to 0.7116 across
          about 0.57 s.

  video9  3 attempts, all unambiguously airborne. A fourth hands-overhead
          event near 2.8 s is the player bending to collect the ball, not a
          shot, and is excluded on purpose.

The clips themselves are ignored by git (assets/videos/**/*.mp4). Re-run this
script to rebuild them; manifest.json records exactly what was produced.

Usage:
    python scripts/build_single_shot_clips.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"
DEST = VIDEO_DIR / "single_shot"

# Lead-in and tail around each attempt, in frames at the source rate. Enough
# to hold the whole motion -- stance, load, lift, release, follow-through,
# landing -- without reaching the neighbouring attempt.
PRE_FRAMES = 30
POST_FRAMES = 50

# (first, last) frame of each confirmed hands-overhead shooting event.
VIDEO8_EVENTS = [(34, 47), (100, 112), (167, 189), (250, 272), (336, 348),
                 (405, 419), (505, 519), (603, 614), (704, 717), (806, 818)]
VIDEO9_EVENTS = [(7, 27), (117, 138), (324, 387)]

VIDEO8_TYPES = ["set"] * 9 + ["jump"]
VIDEO9_TYPES = ["jump"] * 3


def clip_bounds(events, frame_count):
    """Split at the midpoint between attempts so no clip can hold two shots."""
    spans = []
    for i, (first, last) in enumerate(events):
        lo = 0 if i == 0 else (events[i - 1][1] + first) // 2
        hi = frame_count - 1 if i == len(events) - 1 else (last + events[i + 1][0]) // 2
        spans.append((max(lo, first - PRE_FRAMES), min(hi, last + POST_FRAMES)))
    return spans


def cut(video_name, events, types, prefix):
    source = VIDEO_DIR / video_name
    if not source.exists():
        print(f"  SKIP {video_name}: not found")
        return []

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    spans = clip_bounds(events, len(frames))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    manifest = []
    for number, ((start, end), kind) in enumerate(zip(spans, types), 1):
        name = f"{prefix}_shot{number:02d}_{kind}.mp4"
        writer = cv2.VideoWriter(str(DEST / name), fourcc, fps, (width, height))
        for frame in frames[start:end + 1]:
            writer.write(frame)
        writer.release()
        manifest.append({
            "clip": name,
            "source": video_name,
            "expected_type": kind,
            "source_frames": [start, end],
            "duration_s": round((end - start + 1) / fps, 2),
        })
        print(f"  {name}  frames {start}-{end}  "
              f"{(end - start + 1) / fps:.2f}s  expect={kind}")
    return manifest


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print("video8 -> single-shot clips")
    entries = cut("video8.mov", VIDEO8_EVENTS, VIDEO8_TYPES, "video8")
    print("video9 -> single-shot clips")
    entries += cut("video9.mov", VIDEO9_EVENTS, VIDEO9_TYPES, "video9")

    (DEST / "manifest.json").write_text(json.dumps(entries, indent=2))
    print(f"\n{len(entries)} clip(s) written to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
