"""Turn each shot into arrays a model can read, WITHOUT handing it the answer.

    venv/Scripts/python.exe temporal/build_features.py

Writes temporal/data/features.npz, so every later experiment is seconds
instead of a ten-minute pass over the corpus. Re-run this whenever
shots.json changes; nothing else needs to touch video.

WHY THERE IS AN EXCLUSION LIST, AND WHY IT IS THE POINT
------------------------------------------------------
The target is

    takeoff_elevation = max over the event window of
                        (median stance ankle_y  -  ankle_y) / body_height

so three per-frame signals RECONSTRUCT IT EXACTLY: `ankle_image_y` is the
numerator, `body_pixel_height` is the denominator, and `body_rise_ratio` is
very nearly the whole quantity. Hand any of them to a model and it learns
arithmetic. It would score near-perfectly and mean nothing, and on a corpus
this small the score is the only thing anyone would look at.

Excluding them is what makes the task real, and it is also what makes the
model USEFUL. The direct measurement already fails on 17 of 43 shots -- the
camera pans, or the stance was never filmed -- and those shots have joint
angles and wrist trajectories like any other. A model that reads elevation
from HOW THE SHOT WAS MADE can answer where the measurement cannot. A model
that reads it from the ankle position is a slower copy of a function we
already have.

So the model sees joint angles, wrist trajectory relative to the body, and
velocities. It never sees the feet against the floor.

WHY THE SEQUENCES ARE RESAMPLED IN TIME, NOT IN FRAMES
------------------------------------------------------
Footage runs 12-30 fps and includes slow motion, so a fixed number of frames
is a different duration on every clip. Each shot is resampled onto the same
number of samples across its own normalised duration, which puts "a quarter of
the way through the shot" at the same index everywhere.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from utils.quiet import quiet_native_stderr  # noqa: E402

import feedback.shot_tracker as st  # noqa: E402
from scripts.coach_report import analyze_video  # noqa: E402
from temporal.dataset import DATA, MANIFEST_JSON  # noqa: E402
from temporal.extract_shots import SHOTS_JSON  # noqa: E402

FEATURES_NPZ = DATA / "features.npz"
LEAKY_NPZ = DATA / "features_leaky.npz"

# Samples per shot along normalised time. 32 is a choice, not a measurement:
# the median shot here is ~50 frames, so 32 resamples slightly down and does
# not invent resolution that was never recorded.
T_STEPS = 32

# Signals the model is NOT allowed to see, and why. Kept as data rather than
# prose so it can be asserted on: a future channel added to KinematicFeatures
# must be classified deliberately, not inherited by accident.
EXCLUDED = {
    "body_rise_ratio": "IS the target, up to the choice of baseline",
    "ankle_image_y": "the target's numerator",
    "body_pixel_height": "the target's denominator",
    "ankle_baseline_y": "the baseline the target subtracts",
    "hip_x_ratio": (
        "normalised by FRAME width, so it encodes how tightly the clip was "
        "cropped -- a preprocessing artefact, not the player"
    ),
}

# Joint angles, in degrees. NaN where the angle was not valid on that frame;
# never 0.0, which is a real angle.
ANGLE_CHANNELS = ["knee", "hip", "elbow", "shoulder", "trunk"]

# Kinematic channels, all either hip-centred world space (blind to whole-body
# translation, which is exactly why they are safe here) or normalised by the
# player's own on-screen height.
KIN_CHANNELS = [
    "wrist_height_ratio",
    "wrist_above_shoulder_ratio",
    "wrist_height_velocity",
    "knee_angle_delta",
    "total_velocity",
    "wrist_velocity_y",
    "index_velocity_y",
    "nose_velocity_y",
    "index_align_angle",
    "ankle_y_avg",
    "shoulder_y",
    "nose_y",
]

CHANNELS = ANGLE_CHANNELS + KIN_CHANNELS

# The excluded channels, as channels. Used ONLY by `--with-leaks`, which
# builds a deliberately contaminated copy of the dataset.
#
# That copy is not an alternative model, it is the control. A null result --
# "no family beats predicting the mean" -- has two explanations: the data
# carries no signal, or the harness cannot learn. Feeding the harness the
# answer distinguishes them. If it scores near-perfectly with these channels
# and no better than the mean without them, the machinery works and the
# shortage is real.
LEAK_CHANNELS = ["body_rise_ratio", "ankle_image_y", "body_pixel_height"]

_CAPTURED: Dict[int, list] = {}
_orig_score = st.score_shot


def _capture(frames, shot_number, *a, **kw):
    _CAPTURED[shot_number] = list(frames)
    return _orig_score(frames, shot_number, *a, **kw)


st.score_shot = _capture


def _angle(snap, side: str, name: str) -> float:
    key = "trunk" if name == "trunk" else f"{side}_{name}"
    r = snap.angles.get(key) if snap.angles else None
    if r is None or not r.is_valid or r.degrees is None:
        return np.nan
    return float(r.degrees)


def _channel_matrix(frames, leaks: bool = False) -> Optional[np.ndarray]:
    """(n_frames, n_channels) with NaN for anything not measured."""
    if not frames:
        return None
    kin = KIN_CHANNELS + (LEAK_CHANNELS if leaks else [])
    rows: List[List[float]] = []
    for snap in frames:
        f = snap.features
        side = getattr(f, "shooting_side", "right") if f else "right"
        row = [_angle(snap, side, a) for a in ANGLE_CHANNELS]
        for name in kin:
            v = getattr(f, name, None) if f else None
            row.append(np.nan if v is None else float(v))
        rows.append(row)
    return np.asarray(rows, dtype=float)


def _resample(mat: np.ndarray, times_s: np.ndarray, n: int) -> np.ndarray:
    """Resample onto `n` samples across the shot's own duration, in SECONDS.

    NaN gaps are interpolated across rather than dropped: a channel missing for
    three frames in the middle is still that channel, and leaving a hole would
    make the sample length depend on tracking quality.

    A channel that was never measured at all stays NaN and is filled downstream
    -- by a value the model can recognise as "absent", never by a plausible
    number.
    """
    span = times_s[-1] - times_s[0]
    if span <= 0:
        return np.repeat(mat[:1], n, axis=0)
    src = (times_s - times_s[0]) / span
    dst = np.linspace(0.0, 1.0, n)

    out = np.full((n, mat.shape[1]), np.nan)
    for c in range(mat.shape[1]):
        col = mat[:, c]
        ok = ~np.isnan(col)
        if ok.sum() < 2:
            continue
        out[:, c] = np.interp(dst, src[ok], col[ok])
    return out


def _summary(seq: np.ndarray) -> np.ndarray:
    """Per-channel min / max / mean, plus the range. NaN-safe."""
    with np.errstate(all="ignore"):
        lo = np.nanmin(seq, axis=0)
        hi = np.nanmax(seq, axis=0)
        mu = np.nanmean(seq, axis=0)
    return np.concatenate([lo, hi, mu, hi - lo])


def summary_names(channels: List[str]) -> List[str]:
    return (
        [f"{c}_min" for c in channels]
        + [f"{c}_max" for c in channels]
        + [f"{c}_mean" for c in channels]
        + [f"{c}_range" for c in channels]
    )


def build(leaks: bool = False) -> dict:
    shots = json.loads(SHOTS_JSON.read_text(encoding="utf-8"))
    clips = {c["filename"]: c
             for c in json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))}

    by_clip: Dict[str, List[dict]] = {}
    for s in shots:
        by_clip.setdefault(s["clip"], []).append(s)

    seqs, sums, meta = [], [], []
    for i, (clip, rows) in enumerate(sorted(by_clip.items()), 1):
        c = clips[clip]
        # Must match how extract_shots produced this row, or the frames would
        # come from different footage than the target did.
        use_raw = any(r["source"] == "raw_fallback" for r in rows)
        path = PROJECT / (c["path"] if use_raw
                          else (c.get("prepared_path") or c["path"]))
        print(f"[{i:3d}/{len(by_clip)}] {clip[:52]:54s}", end="", flush=True)

        _CAPTURED.clear()
        try:
            with quiet_native_stderr():
                analyze_video(path, height_cm=None, enable_ball=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}")
            continue

        built = 0
        for r in rows:
            frames = _CAPTURED.get(r["shot_number"]) or []
            mat = _channel_matrix(frames, leaks=leaks)
            if mat is None or len(mat) < 4:
                print(f"  shot {r['shot_number']}: too few frames", end="")
                continue
            times = np.array([f.timestamp_ms for f in frames], dtype=float) / 1000.0
            seq = _resample(mat, times, T_STEPS)
            seqs.append(seq)
            sums.append(_summary(seq))
            meta.append({
                "clip": clip,
                "shot_number": r["shot_number"],
                "split": r["split"],
                "group": r["group"],
                "elevation": r["elevation"],
                "filename_label": r["filename_label"],
                "duration_s": r["duration_s"],
                "source": r["source"],
            })
            built += 1
        print(f"  {built} built")

    return {
        "X_seq": np.asarray(seqs, dtype=np.float32),
        "X_sum": np.asarray(sums, dtype=np.float32),
        "meta": meta,
        "channels": CHANNELS + (LEAK_CHANNELS if leaks else []),
    }


def main() -> int:
    leaks = "--with-leaks" in sys.argv
    d = build(leaks=leaks)
    X_seq, X_sum, meta = d["X_seq"], d["X_sum"], d["meta"]
    y = np.array([np.nan if m["elevation"] is None else m["elevation"]
                  for m in meta], dtype=np.float32)

    dest = LEAKY_NPZ if leaks else FEATURES_NPZ
    np.savez_compressed(
        dest,
        X_seq=X_seq,
        X_sum=X_sum,
        y=y,
        meta=json.dumps(meta),
        channels=json.dumps(d["channels"]),
        summary_names=json.dumps(summary_names(d["channels"])),
        excluded=json.dumps(EXCLUDED),
    )

    n_tgt = int(np.sum(~np.isnan(y)))
    print()
    print(f"{len(meta)} shots   sequences {X_seq.shape}   summaries {X_sum.shape}")
    print(f"{n_tgt} carry a target; {len(meta) - n_tgt} do not")
    nan_frac = float(np.isnan(X_seq).mean())
    print(f"{nan_frac:.1%} of sequence cells are NaN (channel never measured "
          "on that shot)")
    if leaks:
        print("\nCONTAMINATED ON PURPOSE -- this is the control, not a model:")
        for k in LEAK_CHANNELS:
            print(f"   {k:22s} INCLUDED")
    else:
        print("\nEXCLUDED from the model, deliberately:")
        for k, why in EXCLUDED.items():
            print(f"   {k:22s} {why}")
    print(f"\nwrote {dest.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
