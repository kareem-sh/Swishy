"""Build the temporal model's dataset: what is in it, what is out, and why.

Run it:

    venv/Scripts/python.exe temporal/dataset.py            # write the manifest
    venv/Scripts/python.exe temporal/dataset.py --report   # print it, write nothing

THE UNIT IS A SHOT, NOT A VIDEO. A clip holding five attempts contributes five
samples. That is why a multi-shot compilation is not excluded for being long --
only for being a duplicate of something already included.

WHY THIS FILE EXISTS RATHER THAN A GLOB OVER assets/videos
----------------------------------------------------------
110 video files were measured. They contain roughly 49 independent shots. The
difference is not tidiness, it is the whole validity of any accuracy number
this project ever reports, and it is invisible from the filenames.

`assets/videos/training/` -- 61 files, the largest directory -- is not footage.
It is SCREEN RECORDINGS of the other videos being played back in a browser.
Confirmed by eye: one frame shows a YouTube player, subscribe button, like and
share icons and the title "20-Minute Shooting Workout with Bjorn Broman", with
the court occupying about a fifth of the frame. 49 of those 61 are re-recordings
of a single gym session that is ALSO present as `video8.mov` and again as the
ten `single_shot/video8_*` cuts.

That one fact explains a measurement that looked like a camera problem: 30% of
sampled frames across the dataset yielded no pose at all, while median landmark
visibility WHEN a pose was found was 0.95. The pose was never jittery. The
player was simply too small in the frame, because most of the frame was
browser chrome. Median subject height is 0.28 of frame height; 60 of 110 clips
sit below 0.30.

Train on that and the model learns letterboxing and one gym, then reports a
high accuracy because it met the same shots again in the test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
DATA = TEMPORAL / "data"

INVENTORY_CSV = DATA / "inventory.csv"
QUALITY_CSV = DATA / "quality.csv"
MANIFEST_JSON = DATA / "manifest.json"

# --------------------------------------------------------------------------
# EXCLUSIONS
#
# Each carries the reason it was excluded, and the reason is kept in the
# manifest. Nothing is dropped silently: a dataset you cannot interrogate is a
# dataset you cannot defend, and "why is that clip missing" is the first
# question anyone will ask.
# --------------------------------------------------------------------------

EXCLUDE_DIRS = {
    "assets/videos/training": (
        "screen recordings of other videos played back in a browser, not "
        "independent footage. 49 of 61 re-record the video8 session, which is "
        "already present as single_shot cuts. Browser chrome occupies most of "
        "the frame, which is why these clips yield no pose on most frames."
    ),
}

EXCLUDE_FILES = {
    "video8.mov": (
        "already cut into the ten single_shot/video8_* clips, which carry "
        "frame-exact windows in manifest.json. Keeping both would duplicate "
        "every shot in it."
    ),
    "video9.mov": (
        "already cut into single_shot/video9_shot01..03."
    ),
    "Salahairballtest.mp4": (
        "a 7 s excerpt of salah_video.mp4 at offset -15.4 s, not a separate "
        "attempt."
    ),
}

# Clips where a SECOND PLAYER of comparable size sits inside the shooter's crop
# for most of the clip, so no crop can isolate one shooter. Measured, not
# assumed: the fraction of frames with an intruder in the crop and the
# intruder's height ratio are both recorded below and both come from quality.csv.
MIN_INTRUDER_FRAC = 0.50
MIN_INTRUDER_SIZE_RATIO = 0.50


@dataclass
class Clip:
    path: str
    filename: str
    directory: str
    player: str
    label: Optional[str]
    duration_s: float
    fps: float
    width: int
    height: int
    dup_group: str
    session_cluster: str
    # quality
    frac_no_pose: Optional[float] = None
    subject_h: Optional[float] = None
    multi_person: bool = False
    intruder_frac: Optional[float] = None
    intruder_size_ratio: Optional[float] = None
    cam_shift: Optional[float] = None
    panel_box: str = ""
    # decisions
    status: str = "include"
    reason: str = ""
    group: str = ""
    split: str = ""
    needs_crop: bool = False
    drop_elevation: bool = False


def _f(row: dict, key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if raw in ("", "None", "nan", "NaN"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def load_clips() -> List[Clip]:
    """Join the inventory and the quality measurements into one record per clip."""
    if not INVENTORY_CSV.exists() or not QUALITY_CSV.exists():
        raise SystemExit(
            f"Missing {INVENTORY_CSV.name} or {QUALITY_CSV.name} in {DATA}. "
            "These are the measured inputs; regenerate them before rebuilding "
            "the manifest."
        )

    quality: Dict[str, dict] = {}
    with QUALITY_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = Path((row.get("video") or "").replace("\\", "/")).name
            quality[key] = row

    clips: List[Clip] = []
    with INVENTORY_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["filename"]
            q = quality.get(name, {})
            label = row.get("label") or ""
            clips.append(
                Clip(
                    path=row["path"].replace("\\", "/"),
                    filename=name,
                    directory=row["directory"].replace("\\", "/"),
                    player=row.get("player") or "",
                    label=None if label in ("", "None") else label,
                    duration_s=_f(row, "duration_s") or 0.0,
                    fps=_f(row, "fps") or 0.0,
                    width=int(_f(row, "width") or 0),
                    height=int(_f(row, "height") or 0),
                    dup_group=row.get("dup_group") or "",
                    session_cluster=row.get("session_cluster") or "",
                    frac_no_pose=_f(q, "frac_no_pose"),
                    subject_h=_f(q, "subject_h_yolo") or _f(q, "subject_h_median"),
                    multi_person=(q.get("multi_person_yolo") or "").lower()
                    in ("true", "1", "yes"),
                    intruder_frac=_f(q, "frac_intruder_in_crop"),
                    intruder_size_ratio=_f(q, "second_person_size_ratio"),
                    cam_shift=_f(q, "cam_shift_median_wpers"),
                    panel_box=q.get("panel_box") or "",
                )
            )
    return clips


def apply_exclusions(clips: List[Clip]) -> None:
    """Mark what is out and why. Order matters only for which reason is shown."""
    for c in clips:
        if c.directory in EXCLUDE_DIRS:
            c.status, c.reason = "exclude", EXCLUDE_DIRS[c.directory]
            continue
        if c.filename in EXCLUDE_FILES:
            c.status, c.reason = "exclude", EXCLUDE_FILES[c.filename]
            continue
        if (
            c.intruder_frac is not None
            and c.intruder_size_ratio is not None
            and c.intruder_frac >= MIN_INTRUDER_FRAC
            and c.intruder_size_ratio >= MIN_INTRUDER_SIZE_RATIO
        ):
            c.status = "exclude"
            c.reason = (
                f"a second player {c.intruder_size_ratio:.2f}x the shooter's "
                f"height is inside the shooter crop on "
                f"{c.intruder_frac * 100:.0f}% of frames, so no crop isolates "
                "one shooter"
            )
            continue


def assign_preprocessing(clips: List[Clip]) -> None:
    """Decide per clip what has to be done to it before features are extracted.

    Both decisions are measurements, not preferences.

    CROP: the shooter is small in frame on most of this corpus. A static crop
    to the union of the shooter's box across the clip, grown 25% and upscaled,
    took pose detection from a median 5/24 frames to 24/24 on eleven of the
    twelve worst clips, including two that had been 0/24.

    The crop is STATIC PER CLIP and must stay that way. `body_rise_ratio` is
    computed in image space as the ankle's displacement from a standing
    baseline, divided by on-screen body height -- so a crop that TRACKS the
    shooter frame by frame subtracts exactly the vertical translation that
    feature exists to measure. The union box moves with nothing.

    ELEVATION DROPPED: on clips where the camera itself pans, `body_rise_ratio`
    cannot separate the player rising from the camera falling. Rather than
    stabilise -- whose residual error would silently bias a feature normalised
    against a baseline accumulated over the whole clip -- the feature is marked
    unavailable and the model sees a missing value. Saying "not measured" is
    always available to us; saying it accurately is the whole discipline.
    """
    for c in clips:
        if c.status != "include":
            continue
        c.needs_crop = (c.subject_h is not None and c.subject_h < 0.45) or bool(
            c.panel_box
        )
        c.drop_elevation = c.cam_shift is not None and c.cam_shift >= 0.02


# The same human under two names in the inventory. Merged explicitly, because
# a leakage check can only catch what it can see, and it cannot see that
# `Klay` and `Klaythompson` are one person.
PLAYER_ALIASES = {
    "klaythompson": "Klay",
}


def _player_of(c: Clip) -> str:
    raw = (c.player or "").strip()
    return PLAYER_ALIASES.get(raw.lower(), raw)


def build_groups(clips: List[Clip]) -> None:
    """Merge clips that share ANY identifier, transitively. Union-find.

    WHY A PRIORITY ORDER IS NOT ENOUGH, AND WHY THIS WAS WRONG FIRST TIME
    ---------------------------------------------------------------------
    The first version picked ONE key per clip -- duplicate set, else session,
    else player -- and that fragments a player across several groups. Measured
    on this corpus: Kevindurant landed in three separate groups, Couch in
    three, Stephcurry in two. Each group was internally sound and the leakage
    check still passed, but only by luck: all of Kevindurant's groups happened
    to fall in `train`. One different choice of test set and the same player
    would have been on both sides with nothing raising.

    A shared identifier is a shared identity, so identity has to be
    transitive. If clip A and clip B share a duplicate set, and B and C share a
    player, then A, B and C are one group -- even though A and C share nothing
    directly. That is what union-find gives and what a priority order cannot.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    included = [c for c in clips if c.status == "include"]
    for c in included:
        me = f"file:{c.filename}"
        find(me)
        for token in (
            f"dup:{c.dup_group}" if c.dup_group else None,
            f"session:{c.session_cluster}" if c.session_cluster else None,
            f"player:{_player_of(c)}" if _player_of(c) else None,
        ):
            if token:
                union(token, me)

    # Name each group after the player in it, so the split is readable and so
    # `TEST_GROUPS` below can be written by hand and stay stable.
    members: Dict[str, List[Clip]] = defaultdict(list)
    for c in included:
        members[find(f"file:{c.filename}")].append(c)
    for root, group in members.items():
        players = sorted({_player_of(c) for c in group if _player_of(c)})
        name = "+".join(players) if players else group[0].filename
        for c in group:
            c.group = name


# Held out and not looked at until the end. Chosen by NAME, deliberately: a
# split that reshuffles whenever the data changes is not a held-out set, it is
# a random number generator with extra steps.
#
# Whole PLAYERS, not whole clips. If a player appears in training the model can
# learn the person -- build, kit, gym, camera angle -- and report an accuracy
# that describes recognition rather than shooting form.
TEST_GROUPS = {"Stephcurry", "Klay"}
VAL_GROUPS = {"Couch2", "Booker"}


def assign_splits(clips: List[Clip]) -> None:
    build_groups(clips)
    for c in clips:
        if c.status != "include":
            continue
        if c.group in TEST_GROUPS:
            c.split = "test"
        elif c.group in VAL_GROUPS:
            c.split = "val"
        else:
            c.split = "train"


def leakage_report(clips: List[Clip]) -> List[str]:
    """Every way a group could straddle a boundary. Loud, not advisory.

    This is the check that decides whether the accuracy number means anything.
    It is not a warning: if it finds something, the split is wrong and the
    number it would produce is fiction.
    """
    problems: List[str] = []
    included = [c for c in clips if c.status == "include"]

    for field_name, label in (
        ("group", "group"),
        ("dup_group", "duplicate set"),
        ("session_cluster", "session"),
        ("player", "player"),
    ):
        spread: Dict[str, set] = defaultdict(set)
        for c in included:
            value = getattr(c, field_name)
            if value:
                spread[value].add(c.split)
        for value, splits in sorted(spread.items()):
            if len(splits) > 1:
                problems.append(
                    f"{label} {value!r} spans {sorted(splits)} -- "
                    "the same footage is on both sides of the split"
                )
    return problems


def summarise(clips: List[Clip]) -> str:
    out: List[str] = []
    inc = [c for c in clips if c.status == "include"]
    exc = [c for c in clips if c.status != "include"]

    out.append(f"{len(clips)} video files measured")
    out.append(f"  included : {len(inc)}")
    out.append(f"  excluded : {len(exc)}")

    by_reason: Dict[str, int] = defaultdict(int)
    for c in exc:
        # First clause of the reason, not first sentence: several reasons carry
        # decimals and filenames, and splitting on "." cut them mid-number.
        by_reason[c.reason.split(",")[0]] += 1
    out.append("\nWHY CLIPS WERE EXCLUDED")
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        out.append(f"  {n:3d}  {reason}")

    out.append("\nSPLIT (by group, never by file)")
    for split in ("train", "val", "test"):
        rows = [c for c in inc if c.split == split]
        groups = sorted({c.group for c in rows})
        hours = sum(c.duration_s for c in rows) / 60.0
        out.append(f"  {split:<6} {len(rows):3d} clips  {len(groups):2d} groups  "
                   f"{hours:5.1f} min")
        for g in groups:
            n = sum(1 for c in rows if c.group == g)
            out.append(f"           {n:3d}  {g}")

    out.append("\nPREPROCESSING")
    out.append(f"  crop+upscale to shooter : {sum(c.needs_crop for c in inc)}")
    out.append(f"  elevation unavailable   : {sum(c.drop_elevation for c in inc)}")

    labelled = [c for c in inc if c.label]
    out.append("\nLABELS ON INCLUDED CLIPS")
    counts: Dict[str, int] = defaultdict(int)
    for c in inc:
        counts[c.label or "unlabelled"] += 1
    for k, v in sorted(counts.items()):
        out.append(f"  {v:3d}  {k}")
    out.append(f"  ({len(labelled)}/{len(inc)} carry a filename label; the rest "
               "must be labelled from the pipeline or by hand)")

    problems = leakage_report(clips)
    out.append("\nLEAKAGE CHECK")
    if problems:
        out.append("  FAILED -- the split is invalid:")
        out.extend(f"    {p}" for p in problems)
    else:
        out.append("  passed: no duplicate set, session or player spans a split")
    return "\n".join(out)


def build() -> List[Clip]:
    clips = load_clips()
    apply_exclusions(clips)
    assign_preprocessing(clips)
    assign_splits(clips)
    return clips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="print the summary and write nothing")
    args = ap.parse_args()

    clips = build()
    print(summarise(clips))

    problems = leakage_report(clips)
    if problems:
        print("\nRefusing to write a manifest for an invalid split.")
        return 1

    if not args.report:
        DATA.mkdir(parents=True, exist_ok=True)
        MANIFEST_JSON.write_text(
            json.dumps([asdict(c) for c in clips], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nwrote {MANIFEST_JSON.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
