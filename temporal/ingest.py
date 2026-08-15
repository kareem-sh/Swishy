"""Measure NEW videos and append them to the inventory and quality tables.

    venv/Scripts/python.exe temporal/ingest.py            # measure what is new
    venv/Scripts/python.exe temporal/ingest.py --dry-run  # look, write nothing

`dataset.py` builds the manifest by joining `inventory.csv` and `quality.csv`.
Both were produced once, by scripts that are not in this repository, so until
now there was NO path from "new videos on disk" to "rows in the dataset". Any
footage added -- downloaded, filmed, or handed over -- simply did not exist as
far as the pipeline was concerned, silently and with no error.

This closes that. It measures only files it has never seen, appends to both
tables, and leaves every existing row untouched.

WHAT IT MEASURES, AND WHY EACH ONE DECIDES SOMETHING
----------------------------------------------------
Nothing here is collected for interest. Each column is consumed by a decision
already in `dataset.py`:

    cam_shift          >= 0.02  ->  elevation is unavailable; the signal cannot
                                    separate a rising player from a falling
                                    camera. Ten of the corpus's 17 missing
                                    targets die here.
    subject_h          <  0.45  ->  crop and upscale before feature extraction
    panel_box     smaller than   ->  letterbox to strip; those pixels are not
                  the frame          image and would inflate the body-height
                                     denominator
    intruder metrics            ->  exclude when a second player of comparable
                                     size sits inside the shooter crop, because
                                     then no crop isolates one shooter
    phash8                      ->  near-duplicate detection, which pins copies
                                     of one shot to ONE side of the split

IDENTITY IS THE ONE THAT CANNOT BE FIXED LATER
----------------------------------------------
Folds are groups, and a group is a person. Filing a new clip of a player
already in the corpus under a new name puts the same human on both sides of
the split, and the leakage check cannot see it -- it can only compare the
names it is given. So `DIRECTORY_PLAYERS` maps a folder to an EXISTING player
where that is who is in it, and it is checked by hand rather than parsed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from temporal.dataset import (  # noqa: E402
    INVENTORY_CSV,
    QUALITY_CSV,
    VIDEO_SUFFIXES,
    parse_label,
    video_dirs,
)

# Folders whose contents are a player ALREADY in the corpus. Written by hand,
# deliberately: guessing identity from a folder name is how the same person
# ends up in two groups, and no automated check can catch that afterwards.
DIRECTORY_PLAYERS = {
    "curry_nba_free_throw_set_Shot": "Stephcurry",
}

# Frames sampled per clip. Fixed count, not a fixed stride, so a 4-second clip
# and a 60-second one cost the same to measure.
SAMPLE_FRAMES = 24

# Two clips whose sampled hashes differ by fewer bits than this are the same
# footage. Compared against every clip already known, not just the new batch.
PHASH_HAMMING_MAX = 12

_YOLO = None


def _yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO

        _YOLO = YOLO("yolov8n.pt")
    return _YOLO


@dataclass
class Measured:
    path: Path
    filename: str
    directory: str
    size_bytes: int
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    codec: str
    player: str
    label: Optional[str]
    label_note: str
    phash8: str
    dup_group: str = ""
    session_cluster: str = ""
    # quality
    frac_no_pose: float = 0.0
    subject_h_yolo: Optional[float] = None
    multi_person_yolo: bool = False
    second_person_size_ratio: Optional[float] = None
    frac_intruder_in_crop: Optional[float] = None
    cam_shift_median_wpers: Optional[float] = None
    panel_box: str = ""
    boxes: list = field(default_factory=list, repr=False)


def _phash(gray: np.ndarray) -> int:
    """64-bit DCT perceptual hash of one frame."""
    small = cv2.resize(gray, (32, 32)).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _panel_box(frames: List[np.ndarray], w: int, h: int) -> str:
    """Content rectangle inside any letterbox / pillarbox.

    Screen recordings carry player chrome and black bars. Those pixels are not
    the image, and leaving them in inflates the frame that every ratio feature
    is measured against.
    """
    if not frames:
        return f"0,0,{w},{h}"
    acc = np.max(np.stack([f for f in frames]), axis=0)
    # A row or column is border if its brightest pixel across the whole clip is
    # still nearly black -- one bright frame is enough to prove it is content.
    rows = np.where(acc.max(axis=1) > 24)[0]
    cols = np.where(acc.max(axis=0) > 24)[0]
    if rows.size == 0 or cols.size == 0:
        return f"0,0,{w},{h}"
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    return f"{x0},{y0},{x1 - x0},{y1 - y0}"


def camera_shift_frames(grays: List[np.ndarray], boxes: List[list], w: int) -> Optional[float]:
    """Median background motion between sampled frames, in frame widths.

    Tracked OUTSIDE the person boxes, because points on a moving player move
    whether or not the camera does -- and it is the camera we are asking
    about. Returns None when there is not enough static texture to tell, which
    is a real answer and must not be reported as zero.
    """
    shifts: List[float] = []
    for i in range(len(grays) - 1):
        a, b = grays[i], grays[i + 1]
        mask = np.full(a.shape, 255, dtype=np.uint8)
        for (x1, y1, x2, y2) in boxes[i]:
            mask[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 0
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01,
                                      minDistance=8, mask=mask)
        if pts is None or len(pts) < 8:
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        if nxt is None or status is None:
            continue
        ok = status.reshape(-1) == 1
        if ok.sum() < 8:
            continue
        d = np.linalg.norm(nxt.reshape(-1, 2)[ok] - pts.reshape(-1, 2)[ok], axis=1)
        shifts.append(float(np.median(d)) / float(w))
    if not shifts:
        return None
    return float(np.median(shifts))


def _intruder_in_crop(boxes: List[list], w: int, h: int) -> Optional[float]:
    """Fraction of frames where ANOTHER person sits inside the shooter's crop.

    Not "frames containing two people" -- that was the first version of this
    and it excluded all nine NBA free throws outright, because a free throw
    has the other team lined up along the lane. They are in the FRAME; they
    are nowhere near the shooter. The exclusion rule downstream asks whether a
    crop can isolate one shooter, so the measurement has to ask the same thing.

    The crop is the shooter's union box grown by a margin, matching
    temporal/preprocess.py. An intruder counts only if it overlaps that box by
    a real fraction of its own area -- a hand or shoulder clipping the edge is
    not a second subject.
    """
    shooters = []
    for bs in boxes:
        if bs:
            shooters.append(max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    if not shooters:
        return None
    arr = np.array(shooters, dtype=float)
    x1, y1 = float(np.percentile(arr[:, 0], 2)), float(np.percentile(arr[:, 1], 2))
    x2, y2 = float(np.percentile(arr[:, 2], 98)), float(np.percentile(arr[:, 3], 98))
    m = (y2 - y1) * 0.25
    cx1, cy1 = max(0.0, x1 - m), max(0.0, y1 - m)
    cx2, cy2 = min(float(w), x2 + m), min(float(h), y2 + m)

    hits = 0
    for bs in boxes:
        if len(bs) < 2:
            continue
        shooter = max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        for b in bs:
            if b is shooter:
                continue
            ox = max(0.0, min(b[2], cx2) - max(b[0], cx1))
            oy = max(0.0, min(b[3], cy2) - max(b[1], cy1))
            area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            if (ox * oy) / area >= 0.5:
                hits += 1
                break
    return round(hits / len(boxes), 3) if boxes else None


def _people(frame: np.ndarray) -> List[list]:
    res = _yolo().predict(frame, classes=[0], verbose=False, conf=0.25)
    out = []
    for r in res:
        for b in r.boxes.xyxy.cpu().numpy():
            out.append([float(v) for v in b[:4]])
    return out


def measure(path: Path) -> Optional[Measured]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if n <= 0 or w <= 0 or h <= 0:
        cap.release()
        return None

    idxs = np.linspace(0, max(n - 1, 0), num=min(SAMPLE_FRAMES, n), dtype=int)
    grays, boxes, hashes = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(g)
        hashes.append(_phash(g))
        boxes.append(_people(frame))
    cap.release()
    if not grays:
        return None

    heights = [sorted([b[3] - b[1] for b in bs], reverse=True) for bs in boxes]
    biggest = [hs[0] / h for hs in heights if hs]
    second = [hs[1] / hs[0] for hs in heights if len(hs) > 1]
    n_pose = sum(1 for bs in boxes if bs)

    rel = path.parent.relative_to(PROJECT).as_posix()
    # The filename first, then the FOLDER. A batch dropped in as 1.mp4, 2.mp4
    # carries no label of its own, but `curry_nba_free_throw_set_Shot` is the
    # owner stating what is inside it -- the same statement, made once for the
    # whole folder instead of once per file.
    label, note = parse_label(path.name)
    if label is None:
        label, note = parse_label(path.parent.name)
    player = DIRECTORY_PLAYERS.get(path.parent.name, "")

    return Measured(
        path=path,
        filename=path.name,
        directory=rel,
        size_bytes=path.stat().st_size,
        fps=round(fps, 2),
        frame_count=n,
        duration_s=round(n / fps, 3) if fps > 0 else 0.0,
        width=w,
        height=h,
        codec="",
        player=player,
        label=label,
        label_note=note,
        phash8="|".join(f"{v:016x}" for v in hashes),
        frac_no_pose=round(1.0 - n_pose / len(boxes), 3) if boxes else 1.0,
        subject_h_yolo=round(float(np.median(biggest)), 3) if biggest else None,
        multi_person_yolo=any(len(bs) > 1 for bs in boxes),
        second_person_size_ratio=(round(float(np.median(second)), 3)
                                  if second else None),
        frac_intruder_in_crop=_intruder_in_crop(boxes, w, h),
        cam_shift_median_wpers=(lambda v: None if v is None else round(v, 5))(
            camera_shift_frames(grays, boxes, w)),
        panel_box=_panel_box(grays, w, h),
        boxes=boxes,
    )


def _known_hashes() -> List[Tuple[str, str, List[int]]]:
    """(filename, dup_group, hashes) for everything already inventoried."""
    out = []
    if not INVENTORY_CSV.exists():
        return out
    with INVENTORY_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("phash8") or "").strip()
            if not raw:
                continue
            try:
                hs = [int(x, 16) for x in raw.split("|") if x]
            except ValueError:
                continue
            out.append((row["filename"], row.get("dup_group") or "", hs))
    return out


def _match_duplicate(m: Measured, known: List[Tuple[str, str, List[int]]]) -> str:
    """Pin a near-duplicate to the group its twin already belongs to.

    Compared against the WHOLE inventory, not just this batch. A re-download of
    a clip already present is the most likely duplicate there is, and pinning
    it is what keeps both copies on one side of the split.
    """
    mine = [int(x, 16) for x in m.phash8.split("|") if x]
    if not mine:
        return ""
    for name, group, hs in known:
        best = min(_hamming(a, b) for a in mine for b in hs)
        if best <= PHASH_HAMMING_MAX:
            return group or f"D_dup_{Path(name).stem[:16]}"
    return ""


def _append(path: Path, rows: List[dict]) -> None:
    """Append, preserving the existing header exactly. Never rewrite old rows."""
    with path.open(encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))
    with path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and report, write nothing")
    args = ap.parse_args()

    if not INVENTORY_CSV.exists() or not QUALITY_CSV.exists():
        raise SystemExit(f"missing {INVENTORY_CSV} or {QUALITY_CSV}")

    seen_sizes = set()
    with INVENTORY_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                seen_sizes.add(int(row["size_bytes"]))
            except (KeyError, ValueError):
                pass

    candidates: List[Path] = []
    for rel in video_dirs():
        d = PROJECT / rel
        if not d.is_dir():
            continue
        # iterdir, not rglob: video_dirs() already lists every subfolder, so
        # recursing counted each file once for its own folder and again for
        # the parent -- 18 candidates from 9 files.
        for p in sorted(d.iterdir()):
            if (p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                    and p.stat().st_size not in seen_sizes):
                candidates.append(p)

    if not candidates:
        print("nothing new: every video on disk is already inventoried")
        return 0

    print(f"{len(candidates)} new video(s)\n")
    known = _known_hashes()
    measured: List[Measured] = []

    for i, p in enumerate(candidates, 1):
        print(f"[{i:3d}/{len(candidates)}] {p.name[:44]:46s}", end="", flush=True)
        m = measure(p)
        if m is None:
            print("  UNREADABLE")
            continue
        m.dup_group = _match_duplicate(m, known)
        measured.append(m)
        print(f"  {m.duration_s:6.1f}s {m.width}x{m.height}"
              f"  subject {m.subject_h_yolo}"
              f"  cam_shift {m.cam_shift_median_wpers}"
              + (f"  DUP of {m.dup_group}" if m.dup_group else "")
              + (f"  player={m.player}" if m.player else "  player=?"))

    print()
    unnamed = [m for m in measured if not m.player]
    if unnamed:
        print(f"{len(unnamed)} clip(s) have NO player, so each becomes its own "
              "group. If any of them is someone already in the corpus, add the "
              "folder to DIRECTORY_PLAYERS before building the manifest -- the "
              "leakage check compares names and cannot see a face:")
        for m in unnamed:
            print(f"    {m.directory}/{m.filename}")
        print()

    pans = [m for m in measured
            if m.cam_shift_median_wpers is not None
            and m.cam_shift_median_wpers >= 0.02]
    print(f"{len(pans)} of {len(measured)} will have NO elevation target "
          "(camera moves too much)")
    for m in pans:
        print(f"    {m.filename[:44]:46s} cam_shift {m.cam_shift_median_wpers}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    _append(INVENTORY_CSV, [
        {"directory": m.directory, "path": str(m.path), "filename": m.filename,
         "size_bytes": m.size_bytes, "size_mb": round(m.size_bytes / 1e6, 3),
         "opened": "True", "fps": m.fps, "frame_count": m.frame_count,
         "duration_s": m.duration_s, "width": m.width, "height": m.height,
         "resolution": f"{m.width}x{m.height}",
         "aspect": round(m.width / m.height, 3) if m.height else "",
         "codec": m.codec, "label": m.label or "None", "label_token": m.label_note,
         "player": m.player, "fps_unusual": "False", "short_lt_0p7s": "False",
         "long_gt_60s": str(m.duration_s > 60), "zero_frames": "False",
         "dup_group": m.dup_group, "session_cluster": m.session_cluster,
         "nearest_other_video": "", "nearest_score": "", "phash8": m.phash8}
        for m in measured])

    _append(QUALITY_CSV, [
        {"video": str(m.path), "name": m.filename, "fps": m.fps,
         "duration_s": m.duration_s, "width": m.width, "height": m.height,
         "frac_no_pose": m.frac_no_pose, "subject_h_yolo": m.subject_h_yolo,
         "subject_h_median": m.subject_h_yolo,
         "multi_person_yolo": str(m.multi_person_yolo),
         "multi_person": str(m.multi_person_yolo),
         "second_person_size_ratio": m.second_person_size_ratio,
         "frac_intruder_in_crop": m.frac_intruder_in_crop,
         "cam_shift_median_wpers": m.cam_shift_median_wpers,
         "panel_box": m.panel_box, "sampled_frames": SAMPLE_FRAMES}
        for m in measured])

    print(f"\nappended {len(measured)} row(s) to "
          f"{INVENTORY_CSV.name} and {QUALITY_CSV.name}")
    print("Next: dataset.py -> preprocess.py -> extract_shots.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def camera_shift(path: Path) -> Optional[float]:
    """Background motion for one clip, without the rest of the measurements.

    Exposed so `collect.py` can apply the same camera limit `dataset.py` does,
    instead of accepting clips whose target the dataset will then drop.
    """
    m = measure(path)
    return None if m is None else m.cam_shift_median_wpers
