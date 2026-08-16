"""Fit every model family, honestly, and say which of them actually works.

    venv/Scripts/python.exe temporal/build_features.py     # once
    venv/Scripts/python.exe temporal/train.py

WHY LEAVE-ONE-GROUP-OUT AND NOT THE VALIDATION SPLIT
----------------------------------------------------
The frozen split gives 16 training targets, 2 validation and 8 test. A
two-sample validation set cannot choose a hyper-parameter: one shot moves it
by half. So model selection happens by LEAVE-ONE-GROUP-OUT cross-validation
over train+val combined -- every development shot gets a prediction from a
model that never saw its group, and there are as many folds as there are
groups instead of one arbitrary split.

The frozen TEST set is not touched until the end, and is used once.

WHY THE KNOB IS CHOSEN INSIDE THE FOLD
--------------------------------------
Picking alpha on the same folds that report the score is how a model gets
credit for a choice made with the answers in hand. Each outer fold runs its
own inner leave-one-group-out over its own training groups, picks the knob
there, and only then predicts the held-out group. It is slower and it is the
only version of this number that means anything.

WHAT A GOOD RESULT LOOKS LIKE HERE
----------------------------------
Not a high score. A constant predictor already classifies most of this corpus
correctly, because elevation is compressed around the 0.12 boundary -- so the
question is never "what accuracy" but "how much better than predicting the
mean", and with 18 development samples the honest answer may be "not
measurably". Four different families are fitted precisely so that agreement
or disagreement between them is visible: if they disagree, the dataset cannot
identify a model, and that is the finding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from temporal.build_features import FEATURES_NPZ  # noqa: E402
from temporal.dataset import DATA  # noqa: E402
from temporal.models import (  # noqa: E402
    KNN,
    BoostedStumps,
    ConstantModel,
    Ridge,
    Standardiser,
    TemporalNet,
)

PREDICTIONS_JSON = DATA / "predictions.json"
RESULTS_JSON = DATA / "results.json"

# The knob, per family. Deliberately coarse: a fine grid searched on 18
# samples finds noise and calls it tuning.
GRIDS = {
    "constant": [{}],
    "ridge": [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0)],
    "knn": [{"k": k} for k in (3, 5, 7)],
    "stumps": [{"rounds": r} for r in (10, 30, 60)],
    "tempnet": [{"epochs": e} for e in (100, 300)],
}
FAMILIES = {
    "constant": ConstantModel,
    "ridge": Ridge,
    "knn": KNN,
    "stumps": BoostedStumps,
    "tempnet": TemporalNet,
}
SEQUENCE_MODELS = {"tempnet"}


def load(path: Path = FEATURES_NPZ) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    y = d["y"]
    keep = ~np.isnan(y)
    return {
        "X_sum": d["X_sum"][keep],
        "X_seq": d["X_seq"][keep],
        "y": y[keep],
        "meta": [m for m, k in zip(meta, keep) if k],
        "summary_names": json.loads(str(d["summary_names"])),
    }


def _fit_predict(fam: str, params: dict, Xtr, ytr, Xte) -> np.ndarray:
    """Standardise on the training fold only, then fit and predict."""
    model = FAMILIES[fam](**params)
    if fam in SEQUENCE_MODELS:
        # (N, T, C): standardise per channel, over samples and time together.
        n_ch = Xtr.shape[2]
        flat = Xtr.reshape(-1, n_ch)
        sc = Standardiser().fit(flat)
        model.fit(sc.transform(flat).reshape(Xtr.shape), ytr)
        return model.predict(sc.transform(Xte.reshape(-1, n_ch)).reshape(Xte.shape))
    sc = Standardiser().fit(Xtr)
    model.fit(sc.transform(Xtr), ytr)
    return model.predict(sc.transform(Xte))


def _groups(meta: List[dict]) -> np.ndarray:
    return np.array([m["group"] for m in meta])


def _pick_knob(fam: str, X, y, g) -> dict:
    """Inner leave-one-group-out. Never sees the outer fold's held-out group."""
    grid = GRIDS[fam]
    if len(grid) == 1:
        return grid[0]
    best, best_mae = grid[0], np.inf
    for params in grid:
        errs = []
        for held in np.unique(g):
            tr, te = g != held, g == held
            if tr.sum() < 3 or te.sum() == 0:
                continue
            try:
                p = _fit_predict(fam, params, X[tr], y[tr], X[te])
            except Exception:                                      # noqa: BLE001
                continue
            errs.extend(np.abs(p - y[te]).tolist())
        if errs and np.mean(errs) < best_mae:
            best, best_mae = params, float(np.mean(errs))
    return best


def cross_validate(fam: str, X, y, g) -> dict:
    """Out-of-fold predictions for every development shot."""
    oof = np.full(len(y), np.nan)
    chosen: Dict[str, dict] = {}
    for held in np.unique(g):
        tr, te = g != held, g == held
        if tr.sum() < 3:
            continue
        params = _pick_knob(fam, X[tr], y[tr], g[tr])
        chosen[held] = params
        oof[te] = _fit_predict(fam, params, X[tr], y[tr], X[te])
    return {"oof": oof, "chosen": chosen}


def _cls(v: float) -> str:
    return "jump" if v >= JUMP_VERTICAL_DISPLACEMENT_RATIO else "set"


def _score(pred: np.ndarray, true: np.ndarray) -> dict:
    ok = ~np.isnan(pred)
    if ok.sum() == 0:
        return {"n": 0, "mae": None, "acc": None}
    return {
        "n": int(ok.sum()),
        "mae": float(np.mean(np.abs(pred[ok] - true[ok]))),
        "acc": float(np.mean([_cls(p) == _cls(t)
                              for p, t in zip(pred[ok], true[ok])])),
    }


def sanity_check(d: dict) -> None:
    """Can this harness learn ANYTHING? Replace the target with a known function.

    A positive control, and the only one that actually works here.

    The first attempt was to hand the models the excluded channels and expect a
    near-perfect score. It did not happen -- the contaminated run scored the
    same as the clean one -- and that is not evidence the harness is broken.
    `takeoff_elevation` is a median stance baseline subtracted from a minimum
    inside a window around the wrist peak. Ridge over per-channel min/max/mean
    cannot express that no matter which channels it is given, so the
    "contaminated" features were never the answer in a form the model could
    read. The control tested the wrong thing.

    A synthetic target tests the right one: y is built from the features by a
    rule the model CAN represent, so recovering it is purely a question of
    whether fit, standardisation, folds and scoring work. If this fails,
    nothing else in this file means anything. If it passes, a null result on
    the real target is about the data.
    """
    X = d["X_sum"]
    g = _groups(d["meta"])
    rng = np.random.default_rng(0)
    sc = Standardiser().fit(X)
    Z = sc.transform(X)
    # A linear rule over three real features, plus noise at a tenth of the
    # signal's spread, so it is learnable but not trivial.
    y_syn = 0.05 * Z[:, 0] - 0.03 * Z[:, 5] + 0.02 * Z[:, 12]
    y_syn = y_syn + rng.normal(0, 0.1 * y_syn.std(), size=len(y_syn))

    print("SANITY CHECK -- synthetic target, linear in three features")
    for fam in ("constant", "ridge", "knn"):
        cv = cross_validate(fam, X, y_syn, g)
        s = _score(cv["oof"], y_syn)
        print(f"  {fam:10s} out-of-fold MAE {s['mae']:.4f}")
    print("  ridge must beat constant here, or the harness is broken.\n")


def main() -> int:
    # `--leaky` runs the identical harness on the contaminated features. It is
    # the control for a null result: if nothing beats the mean on clean
    # features but everything nails it here, the harness works and the data is
    # the limit.
    leaky = "--leaky" in sys.argv
    src = (DATA / "features_leaky.npz") if leaky else FEATURES_NPZ
    if leaky:
        print("=== CONTROL RUN: features CONTAMINATED with the target ===")
        print("=== a high score here proves the harness learns, nothing more ===\n")
    d = load(src)
    if not leaky:
        sanity_check(d)
    meta, y = d["meta"], d["y"]
    g = _groups(meta)
    split = np.array([m["split"] for m in meta])

    dev = split != "test"
    test = split == "test"
    print(f"{len(y)} shots with a target: "
          f"{dev.sum()} development, {test.sum()} test")
    print(f"development groups: {sorted(set(g[dev]))}")
    print(f"test groups:        {sorted(set(g[test]))}")
    print(f"target: mean {y[dev].mean():.4f}  sd {y[dev].std():.4f}  "
          f"range {y[dev].min():.3f}-{y[dev].max():.3f}")
    print()

    results = {}
    preds_out: Dict[str, float] = {}

    for fam in FAMILIES:
        X = d["X_seq"] if fam in SEQUENCE_MODELS else d["X_sum"]
        cv = cross_validate(fam, X[dev], y[dev], g[dev])
        oof = _score(cv["oof"], y[dev])

        params = _pick_knob(fam, X[dev], y[dev], g[dev])
        held = _fit_predict(fam, params, X[dev], y[dev], X[test])
        te = _score(held, y[test])

        results[fam] = {"oof": oof, "test": te, "final_params": params,
                        "per_fold_params": cv["chosen"]}
        print(f"{fam:10s} oof: MAE {oof['mae']:.4f} acc {oof['acc']:.0%} "
              f"(n={oof['n']})   test: MAE {te['mae']:.4f} acc {te['acc']:.0%} "
              f"(n={te['n']})   {params}")

        if fam == "ridge":
            for m, p in zip([m for m, t in zip(meta, test) if t], held):
                preds_out[f"{m['clip']}#{m['shot_number']}"] = float(p)

    base = results["constant"]
    print()
    print("AGAINST THE CONSTANT BASELINE (out-of-fold, the honest column)")
    for fam, r in results.items():
        if fam == "constant":
            continue
        dm = base["oof"]["mae"] - r["oof"]["mae"]
        print(f"  {fam:10s} MAE {'better' if dm > 0 else 'WORSE'} by "
              f"{abs(dm):.4f}   acc {r['oof']['acc'] - base['oof']['acc']:+.0%}")

    if leaky:
        print("\n(control run: nothing written -- these numbers are not a model)")
        return 0
    PREDICTIONS_JSON.write_text(json.dumps(preds_out, indent=2), encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str),
                            encoding="utf-8")
    print(f"\nwrote {PREDICTIONS_JSON.relative_to(PROJECT)} (ridge, held-out set)")
    print(f"wrote {RESULTS_JSON.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
