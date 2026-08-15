"""Find candidate footage online and KEEP ONLY WHAT THE PIPELINE CAN MEASURE.

    venv/Scripts/python.exe temporal/collect.py --search "..." --limit 10
    venv/Scripts/python.exe temporal/collect.py --urls urls.txt

Downloading is the easy half and the worthless half. The corpus already has
40 internet clips and only 26 of their shots carry a target -- the other 17
are lost to panning cameras and to footage that starts after the stance. More
clips of that kind add nothing, so this script's real job is to THROW MOST OF
THEM AWAY, and to say why.

THE ACCEPTANCE TEST IS THE PIPELINE ITSELF
------------------------------------------
Rather than reimplement "is the camera static" and "is the stance visible" as
separate heuristics, a candidate is accepted only if `analyze_video` finds a
shot AND `takeoff_elevation` returns a number for it. That is the exact
condition the dataset needs, measured by the same code that will measure it
later, so a clip that passes here cannot fail there for a reason this stage
was supposed to catch.

It costs a full pipeline run per candidate, which is why the cheap filters
below (duration, resolution, one person) run first.

WHAT IS ACTUALLY SCARCE
-----------------------
Not clips. JUMP shots: 5 of the 26 existing targets clear the 0.12 boundary,
and everything else measures as a set shot, which is why every model collapses
to predicting one class. `--min-elevation` exists so a run can insist on the
side of the boundary that is starving.

And not shots either, but PEOPLE: cross-validation folds are groups, so ten
shots of one new player are worth less than one shot each from ten players.
Each accepted clip is filed under its own group by default.

PROVENANCE
----------
Every accepted clip is recorded with its source URL, title, uploader, licence
and duration in collected.json. A thesis dataset whose origins are not written
down cannot be defended, and "I downloaded it last spring" is not a citation.
Prefer `--creative-commons` where the footage exists: it is the same search
with a licence filter, and it makes the provenance answerable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

TEMPORAL = Path(__file__).resolve().parent
PROJECT = TEMPORAL.parent
sys.path.insert(0, str(PROJECT))

from utils.quiet import quiet_native_stderr  # noqa: E402

import feedback.shot_tracker as st  # noqa: E402
from scripts.coach_report import analyze_video  # noqa: E402
from analysis.engine import MAX_SHOOTING_ELEVATION  # noqa: E402
from shots.classifier import JUMP_VERTICAL_DISPLACEMENT_RATIO  # noqa: E402
from shots.elevation import shooting_event_ms, takeoff_elevation  # noqa: E402
from temporal.dataset import DATA  # noqa: E402

STAGING = DATA / "staging"
ACCEPTED = PROJECT / "assets" / "videos" / "collected"
COLLECTED_JSON = DATA / "collected.json"

# A bound on download and vetting COST, not a judgement about content.
#
# It was 180 s on the reasoning that long videos are compilations with cuts.
# Measured: that rejected 13 of 15 candidates before anything looked at them,
# because useful basketball footage lives inside 5-15 minute tutorials and
# drill videos. The heuristic was deciding what the measurement was there to
# decide -- and `vet` already refuses anything without a measurable elevation,
# so a compilation full of cuts gets thrown out on its merits.
#
# 600 s is where a full pipeline pass stops being worth the wait, and nothing
# more principled than that.
MAX_DURATION_S = 600
MIN_DURATION_S = 3

_CAPTURED: Dict[int, list] = {}
_orig_score = st.score_shot


def _capture(frames, shot_number, *a, **kw):
    _CAPTURED[shot_number] = list(frames)
    return _orig_score(frames, shot_number, *a, **kw)


st.score_shot = _capture


def _ascii(s: Optional[str], n: int = 70) -> str:
    """Console-safe. Windows consoles raise on non-ASCII titles mid-print."""
    if not s:
        return ""
    return s.encode("ascii", "replace").decode("ascii")[:n]


def _elevation_of(frames) -> Optional[float]:
    """The dataset's own target, computed exactly as extract_shots computes it."""
    if not frames:
        return None
    ts = [f.timestamp_ms for f in frames]
    ankles = [f.features.ankle_image_y if f.features else None for f in frames]
    heights = [f.features.body_pixel_height if f.features else 0.0 for f in frames]
    wrists = [f.features.wrist_height_ratio if f.features else None for f in frames]
    event = shooting_event_ms(ts, wrists)
    if event is None:
        return None
    return takeoff_elevation(ts, ankles, heights, wrists, event)


def vet(path: Path, min_elevation: float) -> dict:
    """Run the real pipeline. Accept only what it can actually measure."""
    _CAPTURED.clear()
    try:
        with quiet_native_stderr():
            run = analyze_video(path, height_cm=None, enable_ball=False)
    except Exception as exc:                                       # noqa: BLE001
        return {"ok": False, "why": f"pipeline error: {type(exc).__name__}"}

    if run.is_rejected:
        return {"ok": False,
                "why": f"rejected: {run.rejection.value if run.rejection else '-'}"}
    if not run.shots:
        return {"ok": False, "why": "no shot detected"}

    best: Optional[float] = None
    kept = 0
    for s in run.shots:
        if s.is_rejected:
            continue
        e = _elevation_of(_CAPTURED.get(s.shot_number) or [])
        if e is None:
            continue
        kept += 1
        if best is None or e > best:
            best = e

    if best is None:
        # The two failure modes the existing corpus is full of, and the whole
        # reason this filter exists.
        return {"ok": False,
                "why": "no measurable elevation (camera moves, or the clip "
                       "starts after the stance)"}
    # The ceiling the scoring engine already applies, and which the first
    # version of this file forgot. Without it the filter ACCEPTED two edited
    # compilations reading 0.862 and 0.640 body heights -- nobody jumps two
    # thirds of their own height to shoot, and a cut between camera angles
    # looks exactly like that. A collector whose worst input scores highest is
    # worse than no collector.
    if abs(best) >= MAX_SHOOTING_ELEVATION:
        return {"ok": False,
                "why": f"elevation {best:.3f} exceeds the physical limit "
                       f"({MAX_SHOOTING_ELEVATION}): edited footage or a "
                       "camera cut, not a jump"}
    if best < min_elevation:
        return {"ok": False,
                "why": f"elevation {best:.3f} below the requested "
                       f"{min_elevation:.2f}"}
    return {"ok": True, "elevation": round(float(best), 4),
            "measurable_shots": kept, "total_shots": len(run.shots),
            "type": "jump_shot" if best >= JUMP_VERTICAL_DISPLACEMENT_RATIO
                    else "set_shot"}


def _ydl_opts(dest: Path, cc_only: bool) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        # VIDEO ONLY, and that is not a compromise. Nothing here reads audio,
        # and a video-only stream needs no muxing -- which matters because
        # ffmpeg is not installed, and the merging formats fail outright.
        # Asking for a combined stream instead would silently drop us to
        # whatever progressive rendition YouTube still serves, usually 360p.
        #
        # 720p is the ceiling on purpose: the pipeline normalises by the
        # player's on-screen height, so extra pixels buy nothing and cost
        # download time.
        "format": (
            "bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/"
            "best[height<=720][ext=mp4]/best"
        ),
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "match_filter": yt_filter,
    }
    if cc_only:
        opts["creativecommons"] = True
    return opts


def yt_filter(info, *, incomplete=False):
    d = info.get("duration")
    if d is None:
        return None if incomplete else "no duration"
    if d > MAX_DURATION_S:
        return f"too long ({d}s > {MAX_DURATION_S}s): compilations cut cameras"
    if d < MIN_DURATION_S:
        return f"too short ({d}s)"
    return None


def collect(queries: List[str], urls: List[str], limit: int,
            min_elevation: float, cc_only: bool) -> List[dict]:
    import yt_dlp

    STAGING.mkdir(parents=True, exist_ok=True)
    ACCEPTED.mkdir(parents=True, exist_ok=True)

    targets = list(urls)
    if queries:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "extract_flat": True, "noplaylist": True}) as y:
            for q in queries:
                spec = f"ytsearch{limit}:{q}"
                try:
                    info = y.extract_info(spec, download=False)
                except Exception as exc:                           # noqa: BLE001
                    print(f"  search failed for {_ascii(q, 40)}: "
                          f"{type(exc).__name__}")
                    continue
                for e in info.get("entries", []) or []:
                    if e and e.get("id"):
                        targets.append(f"https://www.youtube.com/watch?v={e['id']}")

    seen, ordered = set(), []
    for u in targets:
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    print(f"{len(ordered)} candidates\n")
    results: List[dict] = []

    for i, url in enumerate(ordered, 1):
        with yt_dlp.YoutubeDL(_ydl_opts(STAGING, cc_only)) as y:
            try:
                info = y.extract_info(url, download=True)
            except Exception as exc:                               # noqa: BLE001
                print(f"[{i:3d}/{len(ordered)}] {url[-11:]}  skipped: "
                      f"{type(exc).__name__} {_ascii(str(exc), 60)}")
                continue
        if info is None:
            print(f"[{i:3d}/{len(ordered)}] {url[-11:]}  filtered out")
            continue

        vid = info.get("id", "unknown")
        found = sorted(STAGING.glob(f"{vid}.*"))
        if not found:
            print(f"[{i:3d}/{len(ordered)}] {vid}  no file written")
            continue
        path = found[0]

        print(f"[{i:3d}/{len(ordered)}] {vid}  {_ascii(info.get('title'), 44):46s}",
              end="", flush=True)
        verdict = vet(path, min_elevation)

        if not verdict["ok"]:
            print(f"  REJECT  {verdict['why']}")
            path.unlink(missing_ok=True)
            results.append({"url": url, "id": vid, "accepted": False,
                            "why": verdict["why"]})
            continue

        # Filed with a group of its own: folds are groups, and two clips of the
        # same uploader are not two independent observations.
        dest = ACCEPTED / f"{vid}_{verdict['type']}{path.suffix}"
        shutil.move(str(path), dest)
        print(f"  ACCEPT  elev {verdict['elevation']:.3f} ({verdict['type']})")
        results.append({
            "url": url, "id": vid, "accepted": True,
            "file": str(dest.relative_to(PROJECT)),
            "title": info.get("title"), "uploader": info.get("uploader"),
            "license": info.get("license"), "duration_s": info.get("duration"),
            "elevation": verdict["elevation"], "type": verdict["type"],
            "measurable_shots": verdict["measurable_shots"],
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", action="append", default=[],
                    help="search query; repeatable")
    ap.add_argument("--urls", type=Path, help="file of URLs, one per line")
    ap.add_argument("--limit", type=int, default=8,
                    help="candidates per query (default 8)")
    ap.add_argument("--min-elevation", type=float, default=0.0,
                    help=f"reject below this; use "
                         f"{JUMP_VERTICAL_DISPLACEMENT_RATIO} to collect only "
                         "jump shots, which is what the corpus is short of")
    ap.add_argument("--creative-commons", action="store_true",
                    help="only CC-licensed results, for cleaner provenance")
    args = ap.parse_args()

    urls: List[str] = []
    if args.urls and args.urls.exists():
        urls = [ln.strip() for ln in args.urls.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")]
    if not args.search and not urls:
        ap.error("give --search or --urls")

    results = collect(args.search, urls, args.limit, args.min_elevation,
                      args.creative_commons)

    ok = [r for r in results if r.get("accepted")]
    print(f"\n{len(ok)} accepted of {len(results)} examined")
    for r in ok:
        print(f"   {r['type']:10s} elev {r['elevation']:.3f}  {r['file']}")

    if results:
        prev = []
        if COLLECTED_JSON.exists():
            prev = json.loads(COLLECTED_JSON.read_text(encoding="utf-8"))
        have = {r["id"] for r in results}
        merged = [p for p in prev if p.get("id") not in have] + results
        COLLECTED_JSON.write_text(json.dumps(merged, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwrote {COLLECTED_JSON.relative_to(PROJECT)} "
              f"({len(merged)} records, provenance included)")

    if ok:
        print("\nNext: re-run dataset.py -> preprocess.py -> extract_shots.py")
        print("Rejected downloads are deleted; only what passed is kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
