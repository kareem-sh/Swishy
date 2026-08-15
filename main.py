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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.coach_report import (  # noqa: E402
    analyze_video,
    print_rejection,
    print_session_summary,
    print_shot,
)

# ==================================
# WHAT TO ANALYSE
# ==================================

VIDEO = "assets/videos/salahRandomshotTest.mov"

# Your height in centimetres, or None. Never estimated from the camera: a
# single lens cannot separate a tall player from a near one. Without it, the
# height-relative metrics are skipped and everything else still runs.
HEIGHT_CM = 178

# "auto" picks the shooting hand from the footage.
SHOOTING_HAND = "auto"

# Ball and rim detection. Slower (loads YOLO) and only affects make/miss.
ENABLE_BALL = True

# After the report is printed, replay the video with the analysis drawn on it.
# The drawing shows the FINAL phase of every frame, which is why it can only
# happen once the whole file has been read.
SHOW_VIDEO = True


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO
    path = Path(video)
    if not path.exists():
        print(f"No such video: {path}")
        return 2

    print(f"\nReading {path.name} ...")
    run = analyze_video(
        path,
        height_cm=HEIGHT_CM,
        shooting_hand=SHOOTING_HAND,
        enable_ball=ENABLE_BALL,
        keep_landmarks=SHOW_VIDEO,
    )

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
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
