"""Render a cleaned copy of every clip the pipeline struggles to see.

    venv/Scripts/python.exe temporal/preprocess.py

The corpus was filmed from the stands and off screens. On the worst clips the
shooter occupies a quarter of the frame height, and MediaPipe -- which is
configured for a single pose (pose/detector.py) -- either misses them or locks
onto whoever is larger. Pose features cannot be fixed downstream of that: a
frame with no landmarks has no angles, no elevation and no phase.

So this stage does two things and refuses to do a third.

CROP TO THE SHOOTER, STATICALLY
-------------------------------
A static box around the shooter, upscaled, gives the detector a subject that
fills the frame. The box is computed ONCE per clip and never moves.

That constraint is the whole design. `body_rise_ratio` and `takeoff_elevation`
measure the ankle's displacement from a standing baseline in IMAGE space, so a
crop that tracked the shooter frame by frame would subtract exactly the
vertical translation those features exist to measure -- and it would do it
silently, producing a plausible number for every jump instead of an obvious
failure. A jump shot would come out looking like a set shot. The union box
moves with nothing, so it cannot remove motion.

REMOVE THE LETTERBOX
--------------------
Several clips are screen recordings with player chrome baked in. The panel box
measured during inventory is intersected with the shooter box, so black bars
never count toward the subject's on-screen height -- which is the denominator
of every ratio feature in the project.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not stabilise panning clips. Residual stabilisation error is a slow
drift, and every ratio feature here is normalised against a baseline gathered
over the clip, so a drift would bias that baseline instead of breaking it --
an error that survives review because the output still looks like data. Those
clips are marked `drop_elevation` in the manifest and their elevation is
reported as missing. Saying "not measured" is always available to us; saying
it accurately is the discipline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from temporal.dataset import DATA, MANIFEST_JSON  # noqa: E402

CLIPS_DIR = DATA / "clips"

# How many frames to sample when locating the shooter. Fixed count, not a
# fixed stride: a 56-second clip and a 4-second clip both get the same budget,
# so the cost of this stage does not depend on how long the footage is.
SAMPLE_FRAMES = 48

# Grow the union box by this much on each side, as a fraction of the box
# HEIGHT -- in both directions, including horizontally.
#
# Scaling the horizontal margin by the box WIDTH, which is the obvious reading,
# cost ten shots. A standing player's box is narrow and tall, so 25% of the
# width of a 133-px box is 33 px per side -- less than a forearm. The arms
# leave that box at release, the wrist landmark goes missing, and the shooting
# event is never found: five of `video8`'s shots stopped being detected at all
# after cropping, having been fine in the raw footage.
#
# Height is the right scale for the same reason the rest of the project
# normalises by it: it is the one dimension of a person that does not change
# with what they are doing with their arms.
BOX_MARGIN = 0.25

# Robust union: instead of min/max over sampled boxes, take these percentiles
# of each edge. One frame where the detector picked a different person -- a
# rebounder walking through, a coach in the foreground -- would otherwise
# stretch the box across the whole court and undo the upscale.
EDGE_LO_PCT = 2.0
EDGE_HI_PCT = 98.0

# Target height for the rendered crop. Upscaling costs nothing in information
# but MediaPipe's detector has a minimum useful subject size, and the council's
# measurement took eleven of the twelve worst clips from a median 5/24 frames
# with a pose to 24/24 at this size.
TARGET_H = 720

_YOLO = None


def _yolo():
    global _YOLO
    if _YOLO is None:
        from ultralytics import YOLO

        _YOLO = YOLO("yolov8n.pt")
    return _YOLO


@dataclass
class Prepared:
    filename: str
    source: str
    output: str
    action: str
    crop: Optional[Tuple[int, int, int, int]]
    scale: float
    note: str


def _panel(clip: dict) -> Optional[Tuple[int, int, int, int]]:
    box = clip.get("panel_box") or ""
    try:
        x, y, w, h = (int(v) for v in box.split(","))
    except ValueError:
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _shooter_boxes(path: Path, n: int) -> List[Tuple[float, float, float, float]]:
    """Largest person box per sampled frame, in pixels.

    Largest, not most central and not most confident. The shooter is the
    subject the camera was pointed at, so they are nearly always the biggest
    person in frame; centrality fails on corner-three footage where the shooter
    stands at the edge, and confidence tracks image quality rather than who
    matters.
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int)

    boxes: List[Tuple[float, float, float, float]] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        res = _yolo().predict(frame, classes=[0], verbose=False, conf=0.25)
        best = None
        best_area = 0.0
        for r in res:
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (float(v) for v in b[:4])
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area, best = area, (x1, y1, x2, y2)
        if best is not None:
            boxes.append(best)
    cap.release()
    return boxes


def _match_aspect(
    x1: float, y1: float, x2: float, y2: float, aspect: float
) -> Tuple[float, float, float, float]:
    """Grow the box -- never shrink it -- until it has the given w/h ratio.

    Growing keeps the shooter inside. Shrinking to fit would crop the subject
    the box was built to contain, which is the one thing this whole stage
    exists to avoid.
    """
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0 or aspect <= 0:
        return x1, y1, x2, y2
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def _union(
    boxes: List[Tuple[float, float, float, float]],
    frame_w: int,
    frame_h: int,
    panel: Optional[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if len(boxes) < 3:
        return None
    arr = np.array(boxes, dtype=float)
    x1 = float(np.percentile(arr[:, 0], EDGE_LO_PCT))
    y1 = float(np.percentile(arr[:, 1], EDGE_LO_PCT))
    x2 = float(np.percentile(arr[:, 2], EDGE_HI_PCT))
    y2 = float(np.percentile(arr[:, 3], EDGE_HI_PCT))

    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return None
    margin = h * BOX_MARGIN
    x1 -= margin
    x2 += margin
    y1 -= margin
    y2 += margin

    # THE CROP MUST KEEP THE SOURCE ASPECT RATIO. This is not cosmetic.
    #
    # Landmarks are normalised per axis: x by frame width, y by frame height.
    # The project then compares the two -- hip travel is a fraction of WIDTH
    # while the body height that normalises everything else is a fraction of
    # HEIGHT. That comparison is only meaningful while the two axes are scaled
    # alike, which an anisotropic crop breaks.
    #
    # Measured: cropping video8_shot04 from 1280x720 to 224x343 scaled x by
    # 5.7 and y by 2.1, so hip travel of 0.084 frame-widths became roughly
    # 0.48 -- past the 0.18 driving threshold. The clip was reclassified LAYUP
    # and rejected, and seven of its siblings went with it. Nothing was wrong
    # with the footage or the classifier; the crop had silently changed the
    # units underneath both.
    # The reference rectangle is the content panel where there is one, and the
    # raw frame otherwise: letterbox pixels are not part of the image and would
    # inflate the subject-height denominator.
    lo_x, lo_y = 0, 0
    hi_x, hi_y = frame_w, frame_h
    if panel:
        px, py, pw, ph = panel
        lo_x, lo_y = max(lo_x, px), max(lo_y, py)
        hi_x, hi_y = min(hi_x, px + pw), min(hi_y, py + ph)
    ref_w, ref_h = hi_x - lo_x, hi_y - lo_y
    if ref_w < 32 or ref_h < 32:
        return None

    x1, y1, x2, y2 = _match_aspect(x1, y1, x2, y2, ref_w / ref_h)

    # Fit by TRANSLATING, not by clamping each edge. Clamping an edge changes
    # one dimension without the other and puts the aspect ratio straight back
    # where it was, which is the bug this is here to prevent.
    w, h = x2 - x1, y2 - y1
    if w >= ref_w or h >= ref_h:
        # Bigger than the space available: take the whole reference rectangle,
        # which already has exactly the right aspect ratio.
        return lo_x, lo_y, ref_w, ref_h
    x1 = min(max(x1, lo_x), hi_x - w)
    y1 = min(max(y1, lo_y), hi_y - h)

    x1, y1 = int(round(x1)), int(round(y1))
    w, h = int(round(w)), int(round(h))
    if w < 32 or h < 32:
        return None
    return x1, y1, w, h


def _render(src: Path, dst: Path, crop: Tuple[int, int, int, int], scale: float) -> bool:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x, y, w, h = crop
    out_w = int(round(w * scale)) // 2 * 2
    out_h = int(round(h * scale)) // 2 * 2
    if out_w < 2 or out_h < 2:
        cap.release()
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    if not writer.isOpened():
        cap.release()
        return False

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # INTER_CUBIC when enlarging: the detector responds to edge sharpness,
        # and INTER_LINEAR softens exactly the limb boundaries it keys on.
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        patch = frame[y : y + h, x : x + w]
        if patch.size == 0:
            continue
        writer.write(cv2.resize(patch, (out_w, out_h), interpolation=interp))
        written += 1
    writer.release()
    cap.release()
    return written > 0


def prepare(clip: dict) -> Prepared:
    src = Path(clip["path"])
    name = clip["filename"]
    panel = _panel(clip)

    if not clip.get("needs_crop"):
        return Prepared(name, str(src), str(src), "passthrough", None, 1.0,
                        "subject already fills the frame")

    boxes = _shooter_boxes(src, SAMPLE_FRAMES)
    crop = _union(boxes, int(clip["width"]), int(clip["height"]), panel)
    if crop is None:
        # Never silently fall back to the full frame: that would record this
        # clip as prepared when nothing was done to it.
        return Prepared(name, str(src), str(src), "failed", None, 1.0,
                        f"no stable shooter box from {len(boxes)} detections")

    scale = TARGET_H / crop[3]
    # An identity crop at unit scale is a lossy re-encode that changes nothing.
    # Happens on long clips where the shooter walks the whole court, so the
    # union box grows to the full frame -- salah_video is 56 seconds of exactly
    # that. Re-encoding it would only cost quality.
    if scale <= 1.0 and crop == (0, 0, int(clip["width"]), int(clip["height"])):
        return Prepared(name, str(src), str(src), "passthrough", None, 1.0,
                        "shooter moves across the whole frame: crop is the frame")

    dst = CLIPS_DIR / name
    if not _render(src, dst, crop, scale):
        return Prepared(name, str(src), str(src), "failed", crop, scale,
                        "encoder wrote no frames")
    return Prepared(name, str(src), str(dst), "cropped", crop, round(scale, 3),
                    f"{crop[2]}x{crop[3]} upscaled x{scale:.2f}")


def main() -> int:
    clips = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    included = [c for c in clips if c["status"] == "include"]
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    done: List[Prepared] = []
    for i, c in enumerate(included, 1):
        print(f"[{i:3d}/{len(included)}] {c['filename'][:52]:54s}", end="", flush=True)
        p = prepare(c)
        print(f"  {p.action:12s} {p.note}")
        done.append(p)

    by_action: dict = {}
    for p in done:
        by_action.setdefault(p.action, []).append(p)
    print()
    for action, rows in sorted(by_action.items()):
        print(f"  {action:12s} {len(rows)}")
    for p in by_action.get("failed", []):
        print(f"    FAILED  {p.filename[:50]:52s} {p.note}")

    # Written back into the manifest so downstream stages read one path and
    # never have to know whether a clip was preprocessed.
    prepared = {p.filename: p.output for p in done}
    for c in clips:
        c["prepared_path"] = prepared.get(c["filename"], c["path"])
    MANIFEST_JSON.write_text(
        json.dumps(clips, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {len(done)} entries and updated "
          f"{MANIFEST_JSON.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
