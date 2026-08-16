"""Find the shots cheaply, then measure only those. Compare against today's path.

    venv/Scripts/python.exe scripts/coarse_to_fine.py                  # the corpus
    venv/Scripts/python.exe scripts/coarse_to_fine.py path/to/clip.mp4 # one file
    venv/Scripts/python.exe scripts/coarse_to_fine.py --stride 6

WHY THIS CAN WORK AT ALL
------------------------
Analysing offline does not mean analysing every frame. Today's path runs
MediaPipe on all of them, and on a five-minute upload that is 9,000 pose
inferences to locate roughly five two-second windows -- about 3% of the
footage. The other 97% is decoded, posed, filtered, angled and scored so that
`segmenter.py` can observe that the hand was down.

So: run pose at a STRIDE to find the peaks, then run the real thing only
inside the windows those peaks define.

WHY THE PEAKS SURVIVE A STRIDE AND THE RULES DO NOT
---------------------------------------------------
`wrist_height_ratio` rises and falls once per shot over ~2 s. Sampling it every
4th frame at 30 fps leaves ~15 samples across that arc, which is plenty to
locate a maximum -- and prominence measures a peak against its own
surroundings, so it degrades gracefully as samples thin out.

The RULES cannot be strided, and this is the part that decides the design:

  - `release` is +/-0.15 s wide. At stride 4 on 30 fps footage that is two
    samples, and the elbow's own extension can fall between them.
  - `_despike` and `_mask_impossible` test a frame against its IMMEDIATE
    neighbours. Widen the gap and a real fast movement starts looking like a
    corrupt frame, and a corrupt one starts looking plausible.
  - `hold_duration_s` is a duration, quantised by the sampling interval.

Hence two passes rather than one cheaper pass.

WHAT THIS DELIBERATELY DOES NOT CLAIM
-------------------------------------
Nothing here is wired into `pipeline.py`. This script measures whether the two
paths AGREE before anything is moved, because the failure mode is not a crash
-- it is a slightly different window producing a slightly different score, on
every upload, invisibly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from mediapipe.tasks.python import vision  # noqa: E402

import feedback.shot_tracker as st  # noqa: E402
from pipeline import ShotAnalysisPipeline  # noqa: E402
from pose import presence  # noqa: E402
from pose.detector import PoseDetector  # noqa: E402
from shots.segmenter import segment  # noqa: E402
from utils.quiet import quiet_native_stderr  # noqa: E402
from utils.video_meta import probe_video  # noqa: E402
from temporal.extract_shots import _elevation  # noqa: E402

# Every Nth frame carries pose in the coarse pass. 4 is a starting point, not a
# result: `--stride` sweeps it, and the agreement table is what picks it.
DEFAULT_STRIDE = 4

# Extra footage either side of a coarse window before the fine pass reads it.
#
# MUST BE AT LEAST `segmenter.WINDOW_PAD_S` (1.2 s), and this is not a margin
# of safety -- it is the condition under which the two paths can agree at all.
#
# The fine pass re-runs `window_for` on its own frames, and `window_for` walks
# outward from the peak up to `WINDOW_PAD_S` looking for where the hand came
# down. Give it less room than that and it cannot reach the bound the full path
# reached; its window closes early, and everything measured RELATIVE TO THE
# WINDOW moves with it.
#
# Measured at 1.0 s: video8 shot 1 opened 366 ms earlier in the fine pass than
# in the full pass, `_stance_indices` recomputed its cutoff from a different
# min/max of the wrist, fewer frames qualified as stance than
# MIN_BASELINE_SAMPLES, and `takeoff_elevation` returned None -- a target
# present in the full path and absent in the fast one. Scores drifted up to 9
# points for the same reason: a moved boundary changes which frame each rule's
# min/max aggregation selects.
#
# 2.5 s also covers `elevation.MAX_LOOKBACK_S`, so the stance the target is
# measured against is inside the window rather than just outside it.
FINE_PAD_S = 2.5

# Two coarse windows closer than this are read as one, so the fine pass never
# analyses the same frames twice under two shot numbers.
MERGE_GAP_S = 0.5

# ------------------------------------------------------ the presence gate
# `pose.presence.PresenceGate` carries the argument for why skipping pose on a
# person-less frame changes neither the pose result nor the ball. What is left
# to MEASURE, and what the `--gate` run below exists for, is whether the claim
# survives contact with real footage: same shots, same elevations, same scores,
# fewer inferences.
#
# Note where the gate is NOT applied: decoding. Every frame is still read and
# still handed to `process_frame`, because `_process_ball` and the ball FSM run
# before the no-pose early return (pipeline.py:615-627) and person-less frames
# are exactly where the rest of the ball's arc is measured.
GATE_PATIENCE_S = presence.DEFAULT_PATIENCE_S
GATE_IDLE_PERIOD_S = presence.DEFAULT_IDLE_PERIOD_S

# Footage fed through the pipeline BEFORE the window, to warm it, and then
# thrown away rather than segmented or scored.
#
# The window alone is not enough to make the two paths agree, and video8 shows
# why in isolation: with `FINE_PAD_S = 2.5` its nine windows come out
# byte-identical to the full path -- same start, same end, same frame count,
# elevation equal to four decimal places -- and shot 3 STILL scored 69 against
# 75. Identical frames, different answer, so the difference cannot be the
# window.
#
# It is state. Two things in the pipeline are accumulated rather than computed
# per frame, and both are younger in a fine pass that starts near the shot:
#
#   - the One Euro filter, whose output depends on its own previous output;
#   - `ankle_image_baseline`, which is seeded on the first credible frame and
#     then blended toward the floor over time.
#
# The full path reaches video8's shot 3 having run 7.7 s of footage through
# both. Feeding the same prefix here and only THEN opening the history gives
# the fine pass the same warmth without letting those frames become a shot.
WARMUP_S = 2.0


# ---------------------------------------------------------------- capture
# Both paths are measured with the same instrument: `score_shot` is wrapped so
# every shot's frames can be re-read afterwards. This is how temporal/
# extract_shots.py already does it, and reusing the mechanism means the
# elevation compared below is computed by exactly one function.
_CAPTURED: dict = {}
_orig_score = st.score_shot


def _capture(frames, shot_number, *a, **kw):
    _CAPTURED[shot_number] = list(frames)
    return _orig_score(frames, shot_number, *a, **kw)


st.score_shot = _capture


@dataclass
class Shot:
    number: int
    t0_ms: int
    t1_ms: int
    score: Optional[int]
    shot_type: str
    elevation: Optional[float]
    frames: int

    @property
    def cls(self) -> str:
        if self.elevation is None:
            return "none"
        return "jump" if self.elevation >= 0.12 else "set"


@dataclass
class Run:
    label: str
    shots: List[Shot] = field(default_factory=list)
    seconds: float = 0.0
    pose_calls: int = 0
    frames_read: int = 0
    skipped: int = 0


def _shots_from(pipe, summaries) -> List[Shot]:
    out: List[Shot] = []
    for s in summaries:
        frames = _CAPTURED.get(s.shot_number) or []
        if not frames:
            continue
        elev, _ = _elevation(frames)
        out.append(Shot(
            number=s.shot_number,
            t0_ms=frames[0].timestamp_ms,
            t1_ms=frames[-1].timestamp_ms,
            score=s.score,
            shot_type=(s.shot_type.value if s.shot_type is not None else "none"),
            elevation=elev,
            frames=len(frames),
        ))
    return out


# ------------------------------------------------------------------ full
def analyse_full(path: Path, fps: float, gated: bool = False) -> Run:
    """Today's path: pose on every frame, then segment.

    With `gated`, the ONE change is that a frame the presence gate declines is
    handed to `process_frame` with `None` where its detection result would go
    -- the same call MediaPipe would have produced on an empty court, reached
    without paying for it. Every frame is still decoded and still passed
    through, so the ball and the landmark array see no difference at all.
    """
    run = Run("full+gate" if gated else "full")
    _CAPTURED.clear()
    t = time.perf_counter()

    pipe = ShotAnalysisPipeline(enable_ball=False)
    pipe.set_fps(fps)
    pipe.enable_offline_segmentation()

    gate = presence.PresenceGate(fps, GATE_PATIENCE_S, GATE_IDLE_PERIOD_S)
    with quiet_native_stderr():
        cap = cv2.VideoCapture(str(path))
        det = PoseDetector(vision.RunningMode.VIDEO)
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(i * 1000.0 / fps)
            result = None
            posed = not gated or gate.should_pose(i)
            if posed:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = det.detect_video_frame(rgb, ts)
                run.pose_calls += 1
            else:
                gate.skip()
            res = pipe.process_frame(result, frame.shape[1], frame.shape[0],
                                     ts, frame)
            if gated and posed:
                gate.observe(i, res.has_pose)
            i += 1
        cap.release()
    run.frames_read = i
    run.skipped = gate.skipped

    summaries = pipe.segment_offline()
    run.shots = _shots_from(pipe, summaries)
    run.seconds = time.perf_counter() - t
    return run


# -------------------------------------------------------- coarse to fine
def _coarse_windows(path: Path, fps: float, stride: int) -> Tuple[List[Tuple[int, int]], int, str]:
    """Candidate (start_ms, end_ms) windows, from pose at 1/stride rate.

    Returns the windows, the number of pose calls spent, and the shooting side
    resolved over the whole clip -- which is handed to the fine pass so the two
    cannot disagree about which arm is shooting for reasons unrelated to the
    stride. `_resolve_shooting_side` accumulates evidence across every frame it
    sees, so the coarse pass sees the whole clip and is the better witness.
    """
    pipe = ShotAnalysisPipeline(enable_ball=False)
    # The history holds only the frames we posed, so its own frame rate is what
    # `segment` must convert seconds against.
    pipe.set_fps(fps / stride)
    pipe.enable_offline_segmentation()

    calls = 0
    with quiet_native_stderr():
        cap = cv2.VideoCapture(str(path))
        det = PoseDetector(vision.RunningMode.VIDEO)
        i = 0
        # The next frame to pose, tracked explicitly rather than as `i % step`.
        # Modulo silently breaks the moment the step changes: going from 4 to
        # 12 at frame 8 means the next multiple of 12 is 12, but from frame 10
        # it is 12 as well -- the gap becomes whatever the arithmetic happens to
        # leave, not the stride that was asked for.
        gate = presence.PresenceGate(
            fps, GATE_PATIENCE_S, GATE_IDLE_PERIOD_S, base_stride=stride
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if gate.should_pose(i):
                # The TRUE timestamp, never a renumbered one: every duration in
                # this project is in seconds, and a strided clock would make
                # every velocity and every window wrong by exactly `stride`.
                ts = int(i * 1000.0 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pipe.process_frame(det.detect_video_frame(rgb, ts),
                                         frame.shape[1], frame.shape[0], ts, frame)
                calls += 1
                gate.observe(i, res.has_pose)
            else:
                gate.skip()
            i += 1
        cap.release()

    history = pipe._offline_history or []
    if not history:
        return [], calls, "right"

    signal = [f.features.wrist_height_ratio if f.features else None for f in history]
    spans = segment(signal, fps / stride)

    pad = int(FINE_PAD_S * 1000)
    raw = [(max(0, history[a].timestamp_ms - pad), history[b].timestamp_ms + pad)
           for a, _, b in spans]

    merged: List[Tuple[int, int]] = []
    for lo, hi in sorted(raw):
        if merged and lo - merged[-1][1] <= MERGE_GAP_S * 1000:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged, calls, pipe._resolved_side


def analyse_coarse_to_fine(path: Path, fps: float, stride: int) -> Run:
    run = Run(f"c2f/{stride}")
    _CAPTURED.clear()
    t = time.perf_counter()

    windows, coarse_calls, side = _coarse_windows(path, fps, stride)
    run.pose_calls = coarse_calls

    number = 0
    with quiet_native_stderr():
        cap = cv2.VideoCapture(str(path))
        det = PoseDetector(vision.RunningMode.VIDEO)
        for lo_ms, hi_ms in windows:
            # A FRESH pipeline per window. Carrying one across a time gap would
            # hand the One Euro filter a dt of several seconds and blend one
            # attempt's ankle baseline into the next.
            pipe = ShotAnalysisPipeline(enable_ball=False, shooting_hand=side)
            pipe.set_fps(fps)

            first = max(0, int(lo_ms * fps / 1000.0))
            last = int(hi_ms * fps / 1000.0)
            warm = max(0, first - int(WARMUP_S * fps))

            cap.set(cv2.CAP_PROP_POS_FRAMES, warm)
            i = warm
            while i <= last:
                ok, frame = cap.read()
                if not ok:
                    break
                # The history opens at the window, not at the warm-up. Frames
                # before `first` move the filter and the ankle baseline and are
                # then invisible: `process_frame` only appends once
                # `enable_offline_segmentation` has been called, so they cannot
                # be segmented into a shot or scored.
                if i == first:
                    pipe.enable_offline_segmentation()
                ts = int(i * 1000.0 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pipe.process_frame(det.detect_video_frame(rgb, ts),
                                   frame.shape[1], frame.shape[0], ts, frame)
                run.pose_calls += 1
                run.frames_read += 1
                i += 1
            if pipe._offline_history is None:
                continue

            _CAPTURED.clear()
            for s in pipe.segment_offline():
                number += 1
                for sh in _shots_from(pipe, [s]):
                    sh.number = number
                    run.shots.append(sh)
        cap.release()

    run.seconds = time.perf_counter() - t
    return run


# ------------------------------------------------------------- comparison
def _pair(a: List[Shot], b: List[Shot]) -> List[Tuple[Optional[Shot], Optional[Shot]]]:
    """Match shots between two runs by TIME OVERLAP, not by index.

    Index matching would silently compare shot 2 of one run against shot 3 of
    the other the moment either found one extra, and every downstream figure
    would then describe two different attempts.
    """
    pairs: List[Tuple[Optional[Shot], Optional[Shot]]] = []
    used = set()
    for x in a:
        best, best_ov = None, 0
        for j, y in enumerate(b):
            if j in used:
                continue
            ov = min(x.t1_ms, y.t1_ms) - max(x.t0_ms, y.t0_ms)
            if ov > best_ov:
                best, best_ov = j, ov
        if best is None:
            pairs.append((x, None))
        else:
            used.add(best)
            pairs.append((x, b[best]))
    pairs += [(None, y) for j, y in enumerate(b) if j not in used]
    return pairs


def _detail(x: Optional[Shot], y: Optional[Shot]) -> dict:
    """One matched pair, side by side.

    A worst-case aggregate names no shot. "Which attempt moved, and by how much
    at each END" is what separates a window edge landing one frame differently
    from the two paths genuinely disagreeing about the shot.
    """
    both = x is not None and y is not None
    return {
        "full_t": [x.t0_ms, x.t1_ms] if x else None,
        "fast_t": [y.t0_ms, y.t1_ms] if y else None,
        "d_start_ms": (y.t0_ms - x.t0_ms) if both else None,
        "d_end_ms": (y.t1_ms - x.t1_ms) if both else None,
        "full_frames": x.frames if x else None,
        "fast_frames": y.frames if y else None,
        "full_score": x.score if x else None,
        "fast_score": y.score if y else None,
        "full_elev": x.elevation if x else None,
        "fast_elev": y.elevation if y else None,
        "full_type": x.shot_type if x else None,
        "fast_type": y.shot_type if y else None,
    }


def compare(name: str, full: Run, fast: Run) -> dict:
    pairs = _pair(full.shots, fast.shots)
    matched = [(x, y) for x, y in pairs if x and y]
    only_full = [x for x, y in pairs if x and not y]
    only_fast = [y for x, y in pairs if y and not x]

    d_elev = [abs(x.elevation - y.elevation) for x, y in matched
              if x.elevation is not None and y.elevation is not None]
    cls_same = sum(1 for x, y in matched if x.cls == y.cls)
    type_same = sum(1 for x, y in matched if x.shot_type == y.shot_type)
    d_score = [abs((x.score or 0) - (y.score or 0)) for x, y in matched
               if x.score is not None and y.score is not None]
    elev_lost = sum(1 for x, y in matched
                    if x.elevation is not None and y.elevation is None)
    elev_gained = sum(1 for x, y in matched
                      if x.elevation is None and y.elevation is not None)

    return {
        "clip": name,
        "shots": [_detail(x, y) for x, y in pairs],
        "full_shots": len(full.shots), "fast_shots": len(fast.shots),
        "matched": len(matched),
        "missed": len(only_full), "extra": len(only_fast),
        "class_agree": cls_same, "type_agree": type_same,
        "elev_lost": elev_lost, "elev_gained": elev_gained,
        "max_d_elev": max(d_elev) if d_elev else None,
        "med_d_elev": sorted(d_elev)[len(d_elev) // 2] if d_elev else None,
        "max_d_score": max(d_score) if d_score else None,
        "full_s": round(full.seconds, 1), "fast_s": round(fast.seconds, 1),
        "speedup": round(full.seconds / fast.seconds, 2) if fast.seconds else None,
        "full_pose": full.pose_calls, "fast_pose": fast.pose_calls,
    }


def _corpus() -> List[Path]:
    """Clips the manifest includes, so the comparison runs on real material."""
    man = PROJECT / "temporal" / "data" / "manifest.json"
    if not man.exists():
        return []
    rows = json.loads(man.read_text(encoding="utf-8"))
    out = []
    for c in rows:
        if c.get("status") != "include":
            continue
        p = PROJECT / c["path"]
        if p.exists():
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="*", type=Path)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N clips of the corpus")
    ap.add_argument("--gate", action="store_true",
                    help="compare the presence gate against the full path, "
                         "instead of coarse-to-fine")
    args = ap.parse_args()

    videos = args.videos or _corpus()
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print("no videos")
        return 2

    rows = []
    for i, v in enumerate(videos, 1):
        meta = probe_video(v)
        if not meta.is_valid:
            print(f"[{i:3d}/{len(videos)}] {v.name[:44]:46s} UNREADABLE")
            continue
        print(f"[{i:3d}/{len(videos)}] {v.name[:44]:46s}", end="", flush=True)
        full = analyse_full(v, meta.fps)
        fast = (analyse_full(v, meta.fps, gated=True) if args.gate
                else analyse_coarse_to_fine(v, meta.fps, args.stride))
        r = compare(v.name, full, fast)
        r["skipped"] = fast.skipped
        r["frames"] = fast.frames_read
        rows.append(r)
        print(f"  full {r['full_shots']} / fast {r['fast_shots']}"
              f"  d_elev {r['max_d_elev']}"
              f"  {r['full_s']}s -> {r['fast_s']}s  x{r['speedup']}"
              + (f"  skipped {fast.skipped}/{fast.frames_read}"
                 if args.gate else ""))

    print()
    print(f"{'':46s} {'shots':>11s} {'match':>6s} {'class':>6s} "
          f"{'maxdE':>7s} {'maxdS':>6s} {'x':>5s}")
    for r in rows:
        de = "   -   " if r["max_d_elev"] is None else f"{r['max_d_elev']:.4f}"
        ds = "-" if r["max_d_score"] is None else str(r["max_d_score"])
        sp = "-" if r["speedup"] is None else f"{r['speedup']:.2f}"
        print(f"{r['clip'][:44]:46s} "
              f"{r['full_shots']:4d} vs {r['fast_shots']:<4d} "
              f"{r['matched']:6d} {r['class_agree']:6d} "
              f"{de:>7s} {ds:>6s} {sp:>5s}")

    tot_full = sum(r["full_shots"] for r in rows)
    tot_fast = sum(r["fast_shots"] for r in rows)
    tot_match = sum(r["matched"] for r in rows)
    tot_cls = sum(r["class_agree"] for r in rows)
    des = [r["max_d_elev"] for r in rows if r["max_d_elev"] is not None]
    dss = [r["max_d_score"] for r in rows if r["max_d_score"] is not None]
    fs = sum(r["full_s"] for r in rows)
    ks = sum(r["fast_s"] for r in rows)

    print()
    print("TOTALS")
    print(f"  shots            full {tot_full}   coarse-to-fine {tot_fast}")
    print(f"  matched by time  {tot_match}")
    print(f"  same jump/set    {tot_cls} of {tot_match}")
    print(f"  elevation lost   {sum(r['elev_lost'] for r in rows)}"
          f"   gained {sum(r['elev_gained'] for r in rows)}")
    print(f"  worst d_elev     {max(des) if des else '-'}"
          f"   (jump/set boundary is 0.12)")
    print(f"  worst d_score    {max(dss) if dss else '-'}")
    print(f"  wall time        {fs:.0f}s -> {ks:.0f}s"
          f"   x{fs / ks:.2f}" if ks else "")
    print(f"  pose calls       {sum(r['full_pose'] for r in rows)} -> "
          f"{sum(r['fast_pose'] for r in rows)}")

    out = PROJECT / "temporal" / "data" / "coarse_to_fine.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
