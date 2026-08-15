"""Turn the included clips into SHOT records -- the actual unit of the dataset.

    venv/Scripts/python.exe temporal/extract_shots.py

A clip is a file. A sample is a shot. `salah_video.mp4` is one clip and five
samples, so a manifest keyed by filename cannot be the training index.

WHAT THIS RECORDS, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
The target is `elevation`, a CONTINUOUS value in the player's own body
heights, measured by `shots.elevation.takeoff_elevation` -- the same function
and the same quantity the shipped classifier uses. See `_elevation` below for
why it is not `max(body_rise_ratio)`, which is what this file measured first
and got wrong.

Shot type is NOT a target here, and that is a decision from the data.

Measured on the cropped clips, elevation runs 0.007 to 0.274 with a median of
0.089, against a jump/set boundary at 0.12. The boundary therefore falls just
above the median, in the densest part of the distribution -- so most of the
corpus sits close enough to it that a small measurement error flips the class.
That is not a badly chosen threshold; it is a continuous quantity being cut
into two, and the cut has to land somewhere.

It also explains the shipped classifier's 74%, with every error in the same
direction: it does not fail randomly, it fails in the overlap.

Training a binary target would spend the model's capacity on a boundary we
chose, teaching it that 0.11 and 0.13 are different classes while 0.01 and
0.11 are the same one. So: predict the number, derive the class downstream
with the same threshold the rest of the project uses. Change the threshold
later and nothing needs retraining.

HOW MUCH OF THIS CORPUS CARRIES A TARGET AT ALL
-----------------------------------------------
Under half. Of 32 detected shots, 19 have a usable elevation; the rest are
missing for two stated reasons, and both are recorded per shot in
`no_elevation_reason` rather than filled in with a zero. That number, not the
clip count, is the size of this dataset.
"""

from __future__ import annotations

import collections
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from utils.quiet import quiet_native_stderr  # noqa: E402

import feedback.shot_tracker as st  # noqa: E402
from scripts.coach_report import analyze_video  # noqa: E402
from analysis.engine import MAX_SHOOTING_ELEVATION  # noqa: E402
from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from shots.elevation import shooting_event_ms, takeoff_elevation  # noqa: E402
from temporal.dataset import DATA, MANIFEST_JSON  # noqa: E402

SHOTS_JSON = DATA / "shots.json"

# The band where the label cannot be trusted, in body heights.
#
# Not a new threshold -- it is the region either side of the project's existing
# 0.12, wide enough to contain every labelled contradiction we measured. Shots
# inside it are kept and trained on; what changes is that they are reported
# separately, so an accuracy figure never averages "did the model learn
# shooting" together with "did it guess our threshold correctly".
AMBIGUOUS_LOW = 0.05
AMBIGUOUS_HIGH = 0.15


def _elevation(frames) -> tuple:
    """The target: elevation at the shooting event, against a stance baseline.

    This deliberately mirrors `_Candidate.takeoff_elevation` rather than taking
    `max(body_rise_ratio)` over the shot. The first version of this file did
    take the max, and it was wrong for reasons the project had already found
    and written down (shot_tracker.py:175):

      - `body_rise_ratio` is measured against a baseline that adapts frame by
        frame, so during a long flight the baseline drifts up and the jump
        shrinks as it happens -- worst in slow motion, which is most of the
        broadcast footage here;
      - a maximum over the whole attempt also catches the player walking out
        to collect the ball, because standing further from the camera lifts
        the feet in the image too.

    The symptom was visible in the extracted data and I misread it: 11 of 12
    filename-vs-pipeline disagreements ran one way, my max reading high against
    the classifier. That was not the classifier being wrong at its own
    threshold. It was this file measuring a different quantity from the one the
    threshold was calibrated on, and `body_rise_ratio` is additionally clamped
    at 0.5, so two shots reported that clamp as if it were a jump.

    `takeoff_elevation` fixes all of it at once: a median stance floor instead
    of an adaptive baseline, a window around the wrist peak instead of the
    whole attempt, no clamp, and None -- never zero -- when the stance was
    never recorded.

    Returns (elevation or None, reason-when-None).
    """
    ts = [f.timestamp_ms for f in frames]
    ankles = [f.features.ankle_image_y if f.features is not None else None for f in frames]
    heights = [f.features.body_pixel_height if f.features is not None else 0.0 for f in frames]
    wrists = [f.features.wrist_height_ratio if f.features is not None else None for f in frames]

    event = shooting_event_ms(ts, wrists)
    if event is None:
        return None, "no wrist peak: the shooting event could not be located"
    elev = takeoff_elevation(ts, ankles, heights, wrists, event)
    if elev is None:
        return None, "no stance before the shot, or the feet were never seen"
    # The scoring engine refuses elevations beyond this as artefacts; a target
    # it would not score is not a target we should train against.
    if abs(elev) >= MAX_SHOOTING_ELEVATION:
        return None, f"elevation {elev:.3f} exceeds the physical limit"
    return float(elev), ""

_CAPTURED = {}
_orig_score = st.score_shot


def _capture(frames, shot_number, *a, **kw):
    _CAPTURED[shot_number] = list(frames)
    return _orig_score(frames, shot_number, *a, **kw)


st.score_shot = _capture


@dataclass
class Shot:
    clip: str
    path: str
    shot_number: int
    split: str
    group: str
    player: str
    # the continuous target
    elevation: Optional[float]
    # derived, never trained against directly
    elevation_class: Optional[str]
    ambiguous_elevation: bool
    # what the pipeline decided, and what the filename claims -- kept apart
    pipeline_type: Optional[str]
    filename_label: Optional[str]
    label_agrees: Optional[bool]
    # why there is no elevation, when there is none -- so "unmeasurable" is
    # never confused with "measured as small"
    no_elevation_reason: str
    # sample geometry
    n_frames: int
    duration_s: float
    fps: float
    score: Optional[int]
    needs_crop: bool
    drop_elevation: bool


def _classify(elev: Optional[float]) -> Optional[str]:
    """Same threshold the rest of the project uses. Derived, never learned."""
    if elev is None:
        return None
    return "jump_shot" if elev >= JUMP_VERTICAL_DISPLACEMENT_RATIO else "set_shot"


def extract(raw: bool = False) -> List[Shot]:
    clips = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    included = [c for c in clips if c["status"] == "include"]
    out: List[Shot] = []

    for i, c in enumerate(included, 1):
        # The preprocessed copy when there is one, the original otherwise.
        # temporal/preprocess.py sets this for every clip precisely so this
        # stage never has to know which clips were cropped.
        #
        # `--raw` forces the originals so the two can be compared directly.
        # Preprocessing is an intervention on the data and has to justify
        # itself against the same measurements as everything else: cropping
        # produced cleaner elevations but detected fewer shots, and only an
        # A/B on identical code says which way that trade goes.
        path = PROJECT / (c["path"] if raw else (c.get("prepared_path") or c["path"]))
        print(f"[{i:3d}/{len(included)}] {c['filename'][:52]:54s}", end="", flush=True)
        _CAPTURED.clear()
        try:
            with quiet_native_stderr():
                run = analyze_video(path, height_cm=None, enable_ball=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}")
            continue

        if run.is_rejected or not run.shots:
            print(f"  no shot ({run.rejection.value if run.rejection else '-'})")
            continue

        kept = 0
        for s in run.shots:
            if s.is_rejected:
                continue
            frames = _CAPTURED.get(s.shot_number) or []
            # None, not 0.0, when nothing was measurable. The whole point.
            elev, why = _elevation(frames)
            # A clip whose camera pans cannot report elevation at all: the
            # signal cannot separate the player rising from the camera falling.
            if c["drop_elevation"]:
                elev, why = None, "camera pans: player rise is not separable"

            ptype = s.shot_type.value if s.shot_type else None
            flabel = c.get("label")
            out.append(
                Shot(
                    clip=c["filename"],
                    path=c["path"],
                    shot_number=s.shot_number,
                    split=c["split"],
                    group=c["group"],
                    player=c["player"],
                    elevation=None if elev is None else round(float(elev), 4),
                    elevation_class=_classify(elev),
                    ambiguous_elevation=(
                        elev is not None and AMBIGUOUS_LOW <= elev <= AMBIGUOUS_HIGH
                    ),
                    pipeline_type=ptype,
                    filename_label=flabel,
                    label_agrees=None if not flabel or not ptype else (flabel == ptype),
                    no_elevation_reason=why,
                    n_frames=len(frames),
                    duration_s=round(len(frames) / (run.fps or 30.0), 3),
                    fps=round(run.fps, 2),
                    score=s.score,
                    needs_crop=bool(c["needs_crop"]),
                    drop_elevation=bool(c["drop_elevation"]),
                )
            )
            kept += 1
        print(f"  {kept} shot(s)")
    return out


def report(shots: List[Shot]) -> str:
    lines: List[str] = []
    lines.append(f"{len(shots)} shots from "
                 f"{len({s.clip for s in shots})} clips")

    for split in ("train", "val", "test"):
        rows = [s for s in shots if s.split == split]
        amb = sum(s.ambiguous_elevation for s in rows)
        noelev = sum(s.elevation is None for s in rows)
        lines.append(
            f"  {split:<6} {len(rows):3d} shots   "
            f"{amb:2d} in the ambiguous band   "
            f"{noelev:2d} with no elevation"
        )

    have = sorted(s.elevation for s in shots if s.elevation is not None)
    if have:
        n = len(have)
        lines.append(
            f"\nelevation  n={n}  min={have[0]:.3f}  p25={have[n//4]:.3f}  "
            f"median={have[n//2]:.3f}  p75={have[(3*n)//4]:.3f}  max={have[-1]:.3f}"
        )

    missing = collections.Counter(
        s.no_elevation_reason for s in shots if s.elevation is None
    )
    if missing:
        lines.append(f"\n{sum(missing.values())} shots have no elevation, because:")
        for reason, n in missing.most_common():
            lines.append(f"    {n:3d}  {reason}")

    disagree = [s for s in shots if s.label_agrees is False]
    lines.append(f"\nfilename label vs pipeline: {len(disagree)} disagreements")
    for s in disagree:
        lines.append(
            f"    {s.clip[:44]:46s} file={s.filename_label:<10s} "
            f"pipeline={s.pipeline_type:<10s} "
            f"elev={s.elevation if s.elevation is not None else float('nan'):.3f}"
        )

    unlabelled = [s for s in shots if not s.filename_label]
    lines.append(f"\n{len(unlabelled)} shots carry no filename label. "
                 "Their elevation_class is DERIVED, not verified -- review "
                 "before treating any of it as ground truth.")
    return "\n".join(lines)


def main() -> int:
    raw = "--raw" in sys.argv
    shots = extract(raw=raw)
    print()
    print(report(shots))
    dest = DATA / ("shots_raw.json" if raw else "shots.json")
    dest.write_text(
        json.dumps([asdict(s) for s in shots], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {dest.relative_to(PROJECT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
