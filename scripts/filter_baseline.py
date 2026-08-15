"""Record what the pipeline scores, so a filter change can be judged.

    python scripts/filter_baseline.py before.json
    ... change config/filter_config.yaml ...
    python scripts/filter_baseline.py after.json --compare before.json

WHY THIS EXISTS
A smoothing parameter does not announce itself. Change it and every angle in
every report shifts a little, scores move, and nothing errors. The only way to
know whether the change helped is to have written down what happened before.

WHAT IT RECORDS
Per video: how many shots were found, and per shot its score, type and rule
outcomes. Detection counts matter as much as scores here -- a filter that
improves an angle but loses a shot has not improved anything.

Ball detection is off: it loads YOLO, costs minutes, and affects only
make/miss, which no smoothing parameter can touch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.coach_report import analyze_video  # noqa: E402

# A spread on purpose: multi-shot and single-shot, set and jump, different
# players and framings. A parameter tuned on one clip is tuned on nothing.
VIDEOS = [
    "assets/videos/salah_video.mp4",
    "assets/videos/video8.mov",
    "assets/videos/video9.mov",
    "assets/videos/video_01_free_throw.mp4",
    "assets/videos/video_07_side_jump_shot.mp4",
    "assets/videos/single_shot/video8_shot01_set.mp4",
    "assets/videos/single_shot/video8_shot05_set.mp4",
    "assets/videos/single_shot/video8_shot10_jump.mp4",
    "assets/videos/single_shot/video9_shot01_jump.mp4",
]


def capture() -> dict:
    out = {}
    for rel in VIDEOS:
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"  skip (missing): {rel}")
            continue
        print(f"  {path.name} ...", flush=True)
        try:
            # No height: keeps the record independent of assumed anthropometry.
            run = analyze_video(path, height_cm=None, enable_ball=False)
        except Exception as exc:                      # noqa: BLE001
            out[rel] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        if run.is_rejected:
            out[rel] = {"rejected": str(run.rejection)}
            continue
        out[rel] = {
            "shots": len(run.shots),
            "discarded": run.discarded_candidates,
            "detail": [
                {
                    "n": s.shot_number,
                    "score": s.score,
                    "type": str(s.shot_type),
                    "passed": s.passed_count,
                    "total": s.total_count,
                    "phase_scores": {p.phase: p.score
                                     for p in (s.phase_scores or [])},
                }
                for s in run.shots
            ],
        }
    return out


def compare(new: dict, old: dict) -> None:
    print("\n" + "=" * 72)
    print("COMPARISON  (old -> new)")
    print("=" * 72)
    moved = kept = 0
    for rel in sorted(set(new) | set(old)):
        a, b = old.get(rel, {}), new.get(rel, {})
        na, nb = a.get("shots"), b.get("shots")
        flag = "" if na == nb else "   <-- SHOT COUNT CHANGED"
        print(f"\n  {Path(rel).name}   shots {na} -> {nb}{flag}")
        da = {d["n"]: d for d in a.get("detail", [])}
        db = {d["n"]: d for d in b.get("detail", [])}
        for n in sorted(set(da) | set(db)):
            x, y = da.get(n), db.get(n)
            if x is None or y is None:
                print(f"    shot {n}: {'ADDED' if x is None else 'REMOVED'}")
                continue
            ds = (y["score"] or 0) - (x["score"] or 0)
            tag = "" if x["type"] == y["type"] else f"  type {x['type']} -> {y['type']}"
            if ds or tag:
                moved += 1
            else:
                kept += 1
            print(f"    shot {n}: score {x['score']} -> {y['score']} "
                  f"({ds:+d}){tag}")
    print(f"\n  {kept} shots unchanged, {moved} moved.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="where to write this capture")
    p.add_argument("--compare", help="an earlier capture to diff against")
    args = p.parse_args()

    print("Capturing pipeline output ...")
    data = capture()
    Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")

    if args.compare:
        old = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        compare(data, old)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
