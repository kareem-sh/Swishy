"""
Run a video through the pipeline and print the coaching report for each shot:
per-phase scores, an overall score, and notes for what is good as well as what
needs changing.

Usage:
    python scripts/coach_report.py assets/videos/video_01_free_throw.mp4
    python scripts/coach_report.py <video> --height-cm 183

--height-cm is OPTIONAL and USER-PROVIDED. Swichy never estimates height from
the camera (see player/profile.py). Without it, height-relative metrics are
skipped and everything else still runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mediapipe.tasks.python import vision  # noqa: E402

from feedback.models import ShotSummary  # noqa: E402
from pipeline import ShotAnalysisPipeline  # noqa: E402
from player.profile import build_player_profile  # noqa: E402
from pose.detector import PoseDetector  # noqa: E402

BAR_WIDTH = 20


def _bar(score: int) -> str:
    filled = int(round(BAR_WIDTH * score / 100.0))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def print_shot(summary: ShotSummary) -> None:
    print()
    print("=" * 72)
    print(f"  SHOT {summary.shot_number}   OVERALL {summary.score}/100  ({summary.grade})")
    print("=" * 72)

    if not summary.phase_scores:
        print("  Not enough of the shot was visible to score it.")
        return

    print("\n  PHASE SCORES")
    print(f"  {'Phase':<22} {'Score':>5}  {'':<20}  Grade")
    print(f"  {'-' * 22} {'-' * 5}  {'-' * 20}  {'-' * 11}")
    for ph in summary.phase_scores:
        print(f"  {ph.label:<22} {ph.score:>3}/100  {_bar(ph.score)}  {ph.grade}")

    print("\n  NOTES BY PHASE")
    for ph in summary.phase_scores:
        print(f"\n  --- {ph.label}  ({ph.score}/100) ---")
        for msg in ph.strengths:
            print(f"    [ON TARGET] {msg}")
        for msg in ph.refinements:
            print(f"    [REFINE]    {msg}")
        for msg in ph.fixes:
            print(f"    [CHANGE]    {msg}")
        for rule in ph.measured:
            value = rule.measured_value
            shown = f"{value:.1f}{rule.unit}" if value is not None else "n/a"
            print(f"    [MEASURED]  {rule.name}: {shown} (not scored)")

    if summary.capture_note:
        print(f"\n  NOTE: {summary.capture_note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="path to a video file")
    parser.add_argument(
        "--height-cm",
        type=float,
        default=None,
        help="player height in centimetres (optional, user-provided only)",
    )
    parser.add_argument("--hand", default="auto", choices=["auto", "left", "right"])
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"No such video: {video_path}")
        return 1

    profile = build_player_profile(height_cm=args.height_cm, shooting_hand=args.hand)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    detector = PoseDetector(vision.RunningMode.VIDEO)
    pipe = ShotAnalysisPipeline(enable_ball=False, player=profile)
    pipe.set_fps(fps)

    print(f"\nVideo        : {video_path.name}  ({fps:.2f} fps)")
    print(f"Player height: {profile.describe_height()}")
    if not profile.has_height:
        print("               -> height-relative metrics will be skipped")

    shots = []
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
            frame,
        )
        if result.shot_summary is not None:
            shots.append(result.shot_summary)
        index += 1
    cap.release()

    trailing = pipe.finalize_session()
    if trailing is not None:
        shots.append(trailing)

    if not shots:
        print("\nNo complete shots detected in this video.")
        return 0

    for summary in shots:
        print_shot(summary)

    print()
    print("=" * 72)
    average = sum(s.score for s in shots) / len(shots)
    print(f"  SESSION: {len(shots)} shot(s), average {average:.0f}/100")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
