"""Analyse a shooting video and print the coaching report.

    python main.py                       # analyses VIDEO below
    python main.py path/to/clip.mp4      # or any file you pass

OFFLINE, NOT LIVE
-----------------
The whole file is read first, and only then is it analysed. That is not a
convenience -- it is what makes the analysis correct.

A shot is a shape in time: the hand rises out of a stance and comes back down.
Deciding frame by frame whether frame 400 belongs to a shot means guessing what
frames 401-460 will do, and the guess needs timeouts, hysteresis and latches to
cover it. Once the file is read, none of that applies: it has all already
happened, and the shot can simply be located.

Live webcam capture is still available through modes/live_stream.py, but it
cannot use this path and is not the product target.

WHAT YOU NEED IN THE FOOTAGE
----------------------------
Start recording with the player STANDING STILL, at least a second before they
begin the shot.

Both measurements depend on it. The shot is found by how far the hand rises
above its own surroundings, which needs footage of those surroundings; and jump
is separated from set shot by how far the feet rise above where they were
standing, which needs to have seen them standing. A clip cut at the moment of
release removes both at once, and the report will say so rather than guess.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.coach_report import (  # noqa: E402
    analyze_video as _analyze_video_run,
    print_rejection,
    print_session_summary,
    print_shot,
)

# ==================================
# WHAT TO ANALYSE
# ==================================

VIDEO = "assets/videos/salahSecondmadeTest.mov"

# Your height in centimetres, or None. Never estimated from the camera: a
# single lens cannot separate a tall player from a near one. Without it, the
# height-relative metrics are skipped and everything else still runs.
HEIGHT_CM = 175

# "auto" picks the shooting hand from the footage.
SHOOTING_HAND = "auto"

# Ball and rim detection. Slower (loads YOLO) and only affects make/miss.
ENABLE_BALL = True

# After the report is printed, replay the video with the analysis drawn on it.
# The drawing shows the FINAL phase of every frame, which is why it can only
# happen once the whole file has been read.
SHOW_VIDEO = True

# Stable machine-readable result from the most recent run. It is intentionally
# overwritten instead of timestamped so an app or script always has one path
# to read.
LATEST_JSON = Path(__file__).resolve().parent / "outputs" / "latest_analysis.json"


def save_latest_analysis(run, output_path: Path = LATEST_JSON) -> Path:
    """Atomically replace the latest-analysis JSON with this run's payload."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(run.to_payload(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        # Normally `replace` has moved the temporary file. Clean up only if a
        # failed write left it behind; the previous valid JSON remains intact.
        if temporary.exists():
            temporary.unlink()
    return path


def analyze_video(
    video_path: str | Path,
    *,
    height_cm: Optional[float] = None,
    rim_height_m: Optional[float] = None,
    shot_xy_m: Optional[Sequence[float]] = None,
    shooting_hand: str = "auto",
    enable_ball: bool = True,
) -> dict:
    """Backend entry point: analyse one upload and return its JSON payload.

    Physical inputs belong to this request. They are passed through memory and
    never written to YAML, which keeps concurrent VPS requests isolated.

    ``shot_xy_m`` uses FIBA half-court coordinates ``[x_m, y_m]``. When
    ``rim_height_m`` or ``shot_xy_m`` is omitted, the local YAML value remains
    the fallback for development runs.
    """
    shot_xy = (
        tuple(shot_xy_m)
        if shot_xy_m is not None
        else None
    )
    run = _analyze_video_run(
        video_path,
        height_cm=height_cm,
        rim_height_m=rim_height_m,
        shot_xy_m=shot_xy,
        shooting_hand=shooting_hand,
        enable_ball=enable_ball,
        keep_landmarks=False,
    )
    return run.to_payload()


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    path = Path(video)
    if not path.exists():
        print(f"No such video: {path}")
        return 2

    print(f"\nReading {path.name} ...")
    run = _analyze_video_run(
        path,
        height_cm=HEIGHT_CM,
        shooting_hand=SHOOTING_HAND,
        enable_ball=ENABLE_BALL,
        keep_landmarks=SHOW_VIDEO,
    )
    latest_json = save_latest_analysis(run)
    print(f"\nLatest JSON: {latest_json}")

    if run.is_rejected:
        print_rejection(run.rejection, run.rejection_detail)
        return 1

    if not run.shots:
        print("\n  No shot was found in this video.")
        if run.no_shot_reason:
            # The specific cause, measured from this clip's own signal, rather
            # than a list of things it might have been.
            print(f"\n  {run.no_shot_reason}")
        return 1

    for shot in run.shots:
        print_shot(shot)
    print_session_summary(run.shots, run.discarded_candidates)

    if SHOW_VIDEO and run.landmarks:
        from modes.replay import replay

        side = run.shots[0].shooting_side if hasattr(run.shots[0], "shooting_side") else "right"
        replay(
            path,
            run.landmarks,
            run.overlay,
            shot_count=len(run.shots),
            shooting_side=side,
            ball_overlay=run.ball_overlay,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
