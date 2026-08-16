"""
Measure whether offline peak detection finds shots better than the live FSM.

MEASUREMENT ONLY. Nothing here is on the production frame path, and nothing
here changes pipeline behaviour. It exists to produce the before/after numbers
that decide whether the replacement is worth shipping.

WHY THIS IS TWO STEPS

Running MediaPipe over a 50 s clip takes minutes. Running `find_peaks` over the
resulting array takes milliseconds. Tuning the second while paying for the
first every time is how three days disappear.

So the signal is extracted ONCE and cached to JSON, and the peak stage reads
the cache. Change a prominence, re-run, get an answer in under a second.

    # once per video (slow)
    python scripts/peak_probe.py extract assets/videos/video8.mov

    # as often as you like (instant)
    python scripts/peak_probe.py peaks --prominence 0.25

    # both, for a first look
    python scripts/peak_probe.py run assets/videos/video8.mov

THE SIGNAL

`wrist_height_ratio` from phase_detection/features.py: the shooting wrist's
height above the hips, divided by the player's own on-screen height. Chosen
because it survives MediaPipe's visibility gate (it is measured in image space,
not the hip-centred world space) and because dividing by body height makes it
invariant to zoom and camera distance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "outputs" / "peak_probe"

# Ground truth for REPORTING ONLY. These numbers are never read by the
# detection code -- they are printed beside the result so a human can see
# whether it was right. Sourced from HANDOFF.md, which records them as visually
# verified from the footage.
#
# `video8_shot09`'s TYPE is disputed (labelled `set`, the user believes `jump`,
# never visually confirmed), but its existence as an attempt is not.
GROUND_TRUTH_SHOTS = {
    "video8.mov": 10,
    "video9.mov": 3,
    "video_01_free_throw.mp4": 2,
    "video_07_side_jump_shot.mp4": 1,
}


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 5)


def _foot_y(image_landmarks, names: tuple) -> Optional[float]:
    """Mean image y of the named foot landmarks, or None if unseen.

    Separate from `_image_metrics` because the ankle cannot answer the
    question these landmarks exist for. Rising onto the balls of the feet
    lifts the ANKLE by 5-8 cm without the foot leaving the floor -- the same
    magnitude as a light jump. The TOE stays down through that rise and only
    lifts when the foot actually leaves the ground, so toe and ankle together
    separate "on tiptoe" from "airborne", which either alone cannot.
    """
    if not image_landmarks:
        return None
    seen = []
    for name in names:
        lm = image_landmarks.get(name)
        if lm is None or float(lm.get("visibility", 0.0)) < 0.5:
            continue
        seen.append(float(lm["y_norm"]))
    return float(np.mean(seen)) if seen else None


def _stillness_threshold() -> float:
    """Speed below which the player counts as standing still, in m/s.

    Matches `still_threshold` used by the existing ankle-baseline updater so
    the two definitions cannot drift apart.
    """
    from utils.config_loader import load_yaml

    cfg = load_yaml("phases.yaml") or {}
    return float((cfg.get("thresholds", {}) or {}).get("ready_max_velocity", 0.2))


# ---------------------------------------------------------------- extraction


def extract(video_path: Path) -> dict:
    """Run pose over the video once and record the per-frame signal."""
    import cv2
    from mediapipe.tasks.python import vision

    from phase_detection.features import _image_metrics
    from pipeline import ShotAnalysisPipeline
    from player.profile import PlayerProfile
    from pose.detector import PoseDetector

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = PoseDetector(vision.RunningMode.VIDEO)
    # Ball disabled: this measures the BODY signal. Ball verification is a
    # separate stage and a separate question.
    pipe = ShotAnalysisPipeline(enable_ball=False, player=PlayerProfile())
    pipe.set_fps(fps)

    frames: List[dict] = []
    fsm_shots = 0
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
            None,
        )

        if result.shot_summary is not None:
            fsm_shots += 1

        features = result.features
        ankle_img_y, _, body_height = _image_metrics(result.image_landmarks)
        frames.append(
            {
                "i": index,
                "t_ms": timestamp_ms,
                # None, never 0.0. A wrist the model could not see is not a
                # wrist at hip height -- that conflation is the `or 0.0` bug
                # HANDOFF.md warns about, and it must not be reintroduced here.
                "wrist": (
                    None
                    if features is None or features.wrist_height_ratio is None
                    else round(float(features.wrist_height_ratio), 5)
                ),
                "ankle_img_y": None if ankle_img_y is None else round(ankle_img_y, 5),
                "toe_img_y": _round_or_none(
                    _foot_y(result.image_landmarks, ("left_foot_index", "right_foot_index"))
                ),
                "heel_img_y": _round_or_none(
                    _foot_y(result.image_landmarks, ("left_heel", "right_heel"))
                ),
                "body_h": round(float(body_height), 5),
                "vel": (
                    0.0 if features is None else round(float(features.total_velocity), 4)
                ),
                "rise_adaptive": (
                    None
                    if features is None or features.body_rise_ratio is None
                    else round(float(features.body_rise_ratio), 5)
                ),
                "phase": result.phase,
                "has_pose": bool(result.has_pose),
            }
        )
        index += 1

    cap.release()

    # What the live FSM produced on the SAME pass, so the comparison is
    # apples-to-apples rather than against a number remembered from a doc.
    if pipe.finalize_session() is not None:
        fsm_shots += 1

    return {
        "video": video_path.name,
        "fps": round(fps, 3),
        "frame_count": index,
        "duration_s": round(index / fps, 2) if fps else None,
        "frames": frames,
        "fsm_shots": fsm_shots,
    }


# ------------------------------------------------------------------- signal


def global_ankle_baseline(
    ankle_y: np.ndarray,
    velocity: np.ndarray,
    body_h: np.ndarray,
    still_threshold: float,
) -> Optional[float]:
    """Where the floor is, decided ONCE for the whole clip.

    The production baseline adapts per frame, so during a long flight it drifts
    upward and the jump measures as smaller than it was. On the slow-motion
    fixture that is the difference between body_rise_ratio 0.031 and ~0.50.

    A median over still frames cannot drift: it is a single number chosen with
    the whole clip visible.
    """
    valid = np.isfinite(ankle_y) & (body_h >= 0.10)
    still = valid & (velocity < still_threshold)
    # Falling back to all valid frames keeps a clip where the player is never
    # still from silently producing "this player never left the floor".
    chosen = still if still.sum() >= 5 else valid
    if chosen.sum() == 0:
        return None
    return float(np.median(ankle_y[chosen]))


def find_peaks_with_prominence(
    signal: np.ndarray, prominence: float, distance: int
) -> tuple:
    """Locate shot candidates. Uses scipy when present, else a numpy fallback."""
    try:
        from scipy.signal import find_peaks, peak_widths

        peaks, props = find_peaks(
            signal, prominence=prominence, distance=max(1, distance)
        )
        if len(peaks) == 0:
            return np.array([], dtype=int), np.array([]), np.array([]), np.array([])
        widths = peak_widths(signal, peaks, rel_height=0.9)
        return peaks, props["prominences"], widths[2], widths[3]
    except ImportError:
        return _fallback_peaks(signal, prominence, distance)


def _fallback_peaks(signal: np.ndarray, prominence: float, distance: int) -> tuple:
    """Prominence by definition, for environments without scipy.

    For each local maximum, walk outward until the signal rises above the peak.
    The highest valley encountered on each side is that side's key col; the
    prominence is the peak minus the HIGHER of the two.
    """
    n = len(signal)
    candidates = [
        i for i in range(1, n - 1)
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]
    ]
    peaks, proms, lefts, rights = [], [], [], []
    for i in candidates:
        left_min = signal[i]
        j = i - 1
        while j >= 0 and signal[j] <= signal[i]:
            left_min = min(left_min, signal[j])
            j -= 1
        right_min = signal[i]
        k = i + 1
        while k < n and signal[k] <= signal[i]:
            right_min = min(right_min, signal[k])
            k += 1
        prom = signal[i] - max(left_min, right_min)
        if prom >= prominence:
            peaks.append(i)
            proms.append(prom)
            cut = signal[i] - prom * 0.9
            lo = i
            while lo > 0 and signal[lo] > cut:
                lo -= 1
            hi = i
            while hi < n - 1 and signal[hi] > cut:
                hi += 1
            lefts.append(lo)
            rights.append(hi)

    # Enforce the minimum gap, keeping the more prominent of any close pair.
    order = np.argsort(proms)[::-1]
    kept: List[int] = []
    for idx in order:
        if all(abs(peaks[idx] - peaks[k]) >= distance for k in kept):
            kept.append(idx)
    kept.sort(key=lambda k: peaks[k])
    return (
        np.array([peaks[k] for k in kept], dtype=int),
        np.array([proms[k] for k in kept]),
        np.array([lefts[k] for k in kept], dtype=float),
        np.array([rights[k] for k in kept], dtype=float),
    )


def _interpolate(signal: List[Optional[float]]) -> np.ndarray:
    """Bridge unseen frames without inventing a height for them.

    A gap becomes a straight line between its two known ends, which is a
    statement about ignorance, not a measurement. Leading and trailing gaps
    take the nearest known value rather than zero -- zero would read as
    "wrist exactly at hip height" and manufacture a peak edge.
    """
    arr = np.array(
        [np.nan if v is None else float(v) for v in signal], dtype=np.float64
    )
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)
    idx = np.arange(len(arr))
    known = ~np.isnan(arr)
    arr[~known] = np.interp(idx[~known], idx[known], arr[known])
    return arr


# -------------------------------------------------------------------- report


def analyse_cache(data: dict, prominence: float, min_gap_s: float) -> dict:
    frames = data["frames"]
    fps = data["fps"]
    signal = _interpolate([f["wrist"] for f in frames])
    ankle = np.array(
        [np.nan if f["ankle_img_y"] is None else f["ankle_img_y"] for f in frames]
    )
    body_h = np.array([f["body_h"] for f in frames])
    vel = np.array([f["vel"] for f in frames])

    distance = max(1, int(round(fps * min_gap_s)))
    peaks, proms, lefts, rights = find_peaks_with_prominence(
        signal, prominence, distance
    )

    baseline = global_ankle_baseline(ankle, vel, body_h, _stillness_threshold())
    rise_global = []
    if baseline is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            raw = (baseline - ankle) / np.where(body_h >= 0.10, body_h, np.nan)
        rise_global = np.clip(np.nan_to_num(raw, nan=0.0), -0.5, 0.5)

    shots = []
    for n, (p, prom, lo, hi) in enumerate(zip(peaks, proms, lefts, rights), 1):
        lo_i, hi_i = int(max(0, lo)), int(min(len(frames) - 1, hi))
        entry = {
            "n": n,
            "peak_t_s": round(frames[p]["t_ms"] / 1000.0, 2),
            "prominence": round(float(prom), 3),
            "wrist_at_peak": round(float(signal[p]), 3),
            "window_s": [
                round(frames[lo_i]["t_ms"] / 1000.0, 2),
                round(frames[hi_i]["t_ms"] / 1000.0, 2),
            ],
            "duration_s": round(
                (frames[hi_i]["t_ms"] - frames[lo_i]["t_ms"]) / 1000.0, 2
            ),
            "rise_adaptive": round(
                float(np.nanmax([frames[i]["rise_adaptive"] or 0.0
                                 for i in range(lo_i, hi_i + 1)])), 3
            ),
        }
        if len(rise_global):
            entry["rise_global"] = round(float(np.max(rise_global[lo_i:hi_i + 1])), 3)
        shots.append(entry)

    return {
        "video": data["video"],
        "fps": fps,
        "duration_s": data["duration_s"],
        "prominence": prominence,
        "min_gap_s": min_gap_s,
        "ankle_baseline_global": None if baseline is None else round(baseline, 4),
        "peaks_found": len(peaks),
        "fsm_shots": data.get("fsm_shots"),
        "expected": GROUND_TRUTH_SHOTS.get(data["video"]),
        "shots": shots,
    }


def report(r: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {r['video']}   ({r['fps']} fps, {r['duration_s']} s)")
    print("=" * 78)
    expected = r["expected"]
    verdict = ""
    if expected is not None:
        verdict = "  <-- MATCH" if expected == r["peaks_found"] else f"  <-- expected {expected}"
    print(f"  prominence={r['prominence']}  min_gap={r['min_gap_s']}s")
    print(f"  peaks found: {r['peaks_found']}{verdict}")
    print(f"  live FSM found: {r['fsm_shots']}   (same pass, same footage)")
    print(f"  global ankle baseline: {r['ankle_baseline_global']}")
    if not r["shots"]:
        print("  (no peaks)")
        return
    print(f"\n  {'#':>3} {'peak':>7} {'prom':>6} {'window':>16} {'dur':>6} "
          f"{'rise_adapt':>11} {'rise_global':>12}")
    for s in r["shots"]:
        window = f"{s['window_s'][0]}-{s['window_s'][1]}"
        print(
            f"  {s['n']:>3} {s['peak_t_s']:>7} {s['prominence']:>6} {window:>16} "
            f"{s['duration_s']:>6} {s['rise_adaptive']:>11} "
            f"{s.get('rise_global', '-'):>12}"
        )


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="run pose once, cache the signal")
    p_extract.add_argument("videos", nargs="+")

    p_peaks = sub.add_parser("peaks", help="detect peaks over cached signals")
    p_peaks.add_argument("--prominence", type=float, default=0.25)
    p_peaks.add_argument("--min-gap", type=float, default=1.0)
    p_peaks.add_argument("--only", nargs="*", default=None)

    p_run = sub.add_parser("run", help="extract then detect, for one look")
    p_run.add_argument("videos", nargs="+")
    p_run.add_argument("--prominence", type=float, default=0.25)
    p_run.add_argument("--min-gap", type=float, default=1.0)

    args = parser.parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.cmd in ("extract", "run"):
        for path in args.videos:
            p = Path(path)
            if not p.exists():
                print(f"skip (missing): {p}")
                continue
            print(f"extracting {p.name} ...", flush=True)
            data = extract(p)
            _cache_path(p.name).write_text(json.dumps(data), encoding="utf-8")
            print(f"  cached {data['frame_count']} frames -> {_cache_path(p.name)}")

    if args.cmd == "extract":
        return 0

    only = getattr(args, "videos", None)
    names = (
        [Path(v).name for v in only]
        if args.cmd == "run"
        else (args.only or [p.stem for p in CACHE_DIR.glob("*.json")])
    )
    for name in names:
        path = _cache_path(name)
        if not path.exists():
            print(f"no cache for {name} -- run `extract` first")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        report(analyse_cache(data, args.prominence, args.min_gap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
