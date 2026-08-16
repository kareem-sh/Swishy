"""A/B the preprocessing stage, per clip, on the measurements that matter.

    venv/Scripts/python.exe temporal/extract_shots.py --raw
    venv/Scripts/python.exe temporal/extract_shots.py
    venv/Scripts/python.exe temporal/compare_preprocessing.py

Cropping and upscaling is an intervention on the data, so it has to justify
itself the same way a threshold change does: by measurement, on both sides,
before and after. The aggregate is not enough -- the first run traded eight
detected shots for cleaner elevations, and an aggregate cannot say whether the
shots lost were the ones that mattered.

Three outcomes are separated because they mean different things:

  RESCUED   no usable elevation raw, one after cropping -- the intended effect
  LOST      a usable elevation raw, none after -- the cost
  CHANGED   usable both ways, but by enough to move the shot across the
            jump/set boundary, which is the only difference that reaches a user
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from temporal.dataset import DATA  # noqa: E402

RAW = DATA / "shots_raw.json"
PREP = DATA / "shots.json"


def _index(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return {
        f"{s['clip']}#{s['shot_number']}": s
        for s in json.loads(path.read_text(encoding="utf-8"))
    }


def _cls(e: Optional[float]) -> str:
    if e is None:
        return "none"
    return "jump" if e >= JUMP_VERTICAL_DISPLACEMENT_RATIO else "set"


def main() -> int:
    raw, prep = _index(RAW), _index(PREP)
    if not raw or not prep:
        print("run extract_shots.py with and without --raw first")
        return 2

    keys = sorted(set(raw) | set(prep))
    rescued, lost, changed, same, appeared, vanished = [], [], [], [], [], []

    for k in keys:
        r, p = raw.get(k), prep.get(k)
        if r is None:
            appeared.append((k, p))
            continue
        if p is None:
            vanished.append((k, r))
            continue
        re_, pe = r["elevation"], p["elevation"]
        if re_ is None and pe is not None:
            rescued.append((k, re_, pe))
        elif re_ is not None and pe is None:
            lost.append((k, re_, pe))
        elif re_ is not None and pe is not None:
            (changed if _cls(re_) != _cls(pe) else same).append((k, re_, pe))

    def _n(v):
        return "none" if v is None else f"{v:.3f}"

    print(f"raw {len(raw)} shots   prepared {len(prep)} shots\n")
    print("SHOT DETECTION")
    print(f"  detected only raw       {len(vanished):3d}   "
          "(cropping cost these shots entirely)")
    for k, r in vanished:
        print(f"      {k[:56]:58s} elev {_n(r['elevation'])}")
    print(f"  detected only prepared  {len(appeared):3d}")
    for k, p in appeared:
        print(f"      {k[:56]:58s} elev {_n(p['elevation'])}")

    print(f"\nELEVATION, on the {len(rescued) + len(lost) + len(changed) + len(same)} "
          "shots detected both ways")
    print(f"  rescued  {len(rescued):3d}   unmeasurable raw, measurable cropped")
    for k, r, p in rescued:
        print(f"      {k[:56]:58s} none -> {_n(p)}")
    print(f"  lost     {len(lost):3d}   measurable raw, unmeasurable cropped")
    for k, r, p in lost:
        print(f"      {k[:56]:58s} {_n(r)} -> none")
    print(f"  class changed {len(changed):3d}")
    for k, r, p in changed:
        print(f"      {k[:56]:58s} {_n(r)} ({_cls(r)}) -> {_n(p)} ({_cls(p)})")
    print(f"  unchanged class {len(same):3d}")

    r_meas = sum(s["elevation"] is not None for s in raw.values())
    p_meas = sum(s["elevation"] is not None for s in prep.values())
    print("\nBOTTOM LINE")
    print(f"  usable targets   raw {r_meas}   prepared {p_meas}")
    print("  Preprocessing is worth keeping only if it adds usable targets or "
          "corrects wrong ones. If it does neither, the raw clips are the "
          "honest input and this stage should be dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
