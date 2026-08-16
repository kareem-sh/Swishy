"""Score a model's predictions in a way that cannot flatter them.

    venv/Scripts/python.exe temporal/evaluate.py predictions.json

Model-agnostic on purpose: this file knows nothing about architectures. It
takes {shot_key: predicted_elevation} and the frozen split, and reports what
the prediction is worth. Written BEFORE any model exists, so no choice here
was made after seeing a result.

Shot keys are "<clip>#<shot_number>", matching temporal/data/shots.json.

WHY THIS IS NOT JUST AN ACCURACY NUMBER
---------------------------------------
The test set is nine shots from two groups, and fewer once unmeasurable ones
are removed. On a sample that size a single shot moves "accuracy" by eleven
points, so a bare percentage is not a measurement -- it is a coin flip with a
decimal point. Four things are therefore reported alongside it, and each one
exists to catch a specific way this dataset could produce a good number for a
bad reason.

1. THE BASELINE THAT PREDICTS THE TRAINING MEAN
   Elevation on this corpus is compressed: measured median 0.084 against a
   0.12 class boundary. A constant predictor is therefore already close to
   most shots and already classifies most of them "correctly". A model that
   does not beat it by a clear margin has learned the mean, not the movement,
   and MAE alone will not say so.

2. THE AMBIGUOUS BAND, HELD APART
   Labels contradict each other between 0.05 and 0.15 -- a `set` at 0.114 and
   a `jump` at 0.075 sit either side of the boundary at almost the same
   height. Averaging those in with the clear cases mixes "did it learn
   shooting" together with "did it guess our threshold", and the second is
   not a skill. Both bands are reported; neither is dropped.

3. THE TRAIN-TEST GAP
   Twenty-odd measurable samples will be memorised by anything with capacity.
   Near-zero training error next to ordinary test error is the signature, and
   it is only visible if training error is printed too.

4. A BOOTSTRAP INTERVAL OVER GROUPS, NOT SHOTS
   Shots from one clip are not independent -- same player, same session, same
   camera. Resampling shots would treat five of Salah's shots as five
   independent observations and produce an interval far too narrow.
   Resampling GROUPS keeps the unit of independence honest, and the resulting
   interval is wide. That width is the finding, not a defect in the estimate.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from temporal.extract_shots import AMBIGUOUS_HIGH, AMBIGUOUS_LOW, SHOTS_JSON  # noqa: E402

BOOTSTRAP_ROUNDS = 2000
BOOTSTRAP_SEED = 20260815


def key(shot: dict) -> str:
    return f"{shot['clip']}#{shot['shot_number']}"


def _load(split: str) -> List[dict]:
    shots = json.loads(SHOTS_JSON.read_text(encoding="utf-8"))
    return [
        s for s in shots
        if s["split"] == split and s["elevation"] is not None
    ]


def _cls(elev: float) -> str:
    return "jump_shot" if elev >= JUMP_VERTICAL_DISPLACEMENT_RATIO else "set_shot"


def _mae(pairs: Sequence[tuple]) -> Optional[float]:
    if not pairs:
        return None
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


def _acc(pairs: Sequence[tuple]) -> Optional[float]:
    if not pairs:
        return None
    return sum(_cls(p) == _cls(t) for p, t in pairs) / len(pairs)


def _bootstrap_by_group(rows: List[dict], preds: Dict[str, float]) -> Optional[tuple]:
    """95% interval on classification accuracy, resampling GROUPS with replacement."""
    groups: Dict[str, List[tuple]] = {}
    for s in rows:
        if key(s) in preds:
            groups.setdefault(s["group"], []).append((preds[key(s)], s["elevation"]))
    names = list(groups)
    if len(names) < 2:
        return None

    rng = random.Random(BOOTSTRAP_SEED)
    accs: List[float] = []
    for _ in range(BOOTSTRAP_ROUNDS):
        pairs: List[tuple] = []
        for _ in names:
            pairs.extend(groups[rng.choice(names)])
        a = _acc(pairs)
        if a is not None:
            accs.append(a)
    if not accs:
        return None
    accs.sort()
    lo = accs[int(0.025 * len(accs))]
    hi = accs[min(int(0.975 * len(accs)), len(accs) - 1)]
    return lo, hi


def evaluate(preds: Dict[str, float]) -> str:
    train = _load("train")
    out: List[str] = []

    if not train:
        return "no measurable training shots: nothing to compare against"
    baseline = sum(s["elevation"] for s in train) / len(train)
    out.append(f"training-mean baseline = {baseline:.4f} body heights "
               f"(from {len(train)} measurable train shots)")

    for split in ("train", "val", "test"):
        rows = _load(split)
        have = [s for s in rows if key(s) in preds]
        missing = len(rows) - len(have)

        out.append("")
        out.append(f"{split.upper()}  {len(rows)} shots with a target, "
                   f"{len(have)} predicted")
        if not have:
            continue

        pairs = [(preds[key(s)], s["elevation"]) for s in have]
        base = [(baseline, s["elevation"]) for s in have]

        m, bm = _mae(pairs), _mae(base)
        # Accuracy is over EVERY shot with a target, not only the predicted
        # ones. A model that declines to predict has not got the shot right,
        # and scoring it on the subset it chose to answer is how a model with
        # a confidence gate reports a number it did not earn. MAE has no
        # defined value for a missing prediction, so it stays on the subset
        # and says so.
        a = None if not rows else (
            sum(_cls(p) == _cls(t) for p, t in pairs) / len(rows)
        )
        ba = _acc(base)
        verdict = "TIES" if abs(bm - m) < 5e-5 else ("BEATS" if m < bm else "LOSES TO")
        out.append(f"  MAE       {m:.4f}    baseline {bm:.4f}    "
                   f"{verdict} baseline by {abs(bm - m):.4f}"
                   + (f"   (on the {len(have)} predicted)" if missing else ""))
        out.append(f"  accuracy  {a:.1%}      baseline {ba:.1%}"
                   + (f"   ({missing} unpredicted counted wrong)" if missing else "")
                   + ("   <- a constant predictor already does this"
                      if a is not None and ba is not None and a <= ba else ""))

        clear = [(p, t) for p, t in pairs
                 if not (AMBIGUOUS_LOW <= t <= AMBIGUOUS_HIGH)]
        amb = [(p, t) for p, t in pairs
               if AMBIGUOUS_LOW <= t <= AMBIGUOUS_HIGH]
        ca, aa = _acc(clear), _acc(amb)
        out.append(f"    clear band      n={len(clear):2d}  "
                   + (f"acc {ca:.1%}" if ca is not None else "no samples"))
        out.append(f"    ambiguous band  n={len(amb):2d}  "
                   + (f"acc {aa:.1%}" if aa is not None else "no samples")
                   + "   (label reliability is the limit here, not the model)")

        ci = _bootstrap_by_group(have, preds)
        if ci:
            out.append(f"  95% interval on accuracy, resampling groups: "
                       f"[{ci[0]:.0%}, {ci[1]:.0%}]")
        else:
            gs = {s['group'] for s in have}
            out.append(f"  no interval: {len(gs)} group(s) in this split, "
                       "and one group cannot be resampled")

        per: Dict[str, List[tuple]] = {}
        for s in have:
            per.setdefault(s["group"], []).append((preds[key(s)], s["elevation"]))
        for g, gp in sorted(per.items()):
            out.append(f"    {g:<14s} n={len(gp):2d}  MAE {_mae(gp):.4f}  "
                       f"acc {_acc(gp):.0%}")

    tr = [(preds[key(s)], s["elevation"]) for s in _load("train") if key(s) in preds]
    te = [(preds[key(s)], s["elevation"]) for s in _load("test") if key(s) in preds]
    tm, tem = _mae(tr), _mae(te)
    if tm is not None and tem is not None:
        out.append("")
        out.append(f"MEMORISATION  train MAE {tm:.4f} vs test MAE {tem:.4f}"
                   f"  (ratio {tem / tm:.1f}x)" if tm > 0 else
                   "MEMORISATION  train MAE is exactly zero -- memorised")
        if tm > 0 and tem / tm > 2.0:
            out.append("  the model fits training shots more than twice as well "
                       "as unseen ones: treat the test figure as the real one "
                       "and the training figure as recall")
    return "\n".join(out)


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        print("\nno predictions file given. Expected JSON: "
              '{"clip.mp4#1": 0.13, ...}')
        return 2
    preds = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    preds = {k: float(v) for k, v in preds.items()}
    print(evaluate(preds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
