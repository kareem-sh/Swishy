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
from shots.types import RejectionReason, describe  # noqa: E402
from utils.video_meta import probe_video  # noqa: E402

BAR_WIDTH = 20

# Below this share of frames containing a detectable person, the clip is not
# footage of someone shooting. Catches non-basketball input outright.
MIN_POSE_FRAME_SHARE = 0.30

REJECTION_TEXT = {
    RejectionReason.INVALID_VIDEO: "The video could not be analysed.",
    RejectionReason.NO_PERSON_DETECTED: (
        "No person was detected in this video, so there is nothing to coach."
    ),
    RejectionReason.NO_SHOOTING_EVENT: (
        "A person was detected, but no shooting attempt was found."
    ),
    RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET: (
        "This shot was recognised, but Swichy does not analyse this shot type yet."
    ),
}


def print_rejection(reason: RejectionReason, detail: str = "") -> None:
    print()
    print("=" * 72)
    print(f"  NOT ANALYSED — {reason.value}")
    print("=" * 72)
    print(f"  {REJECTION_TEXT.get(reason, 'This video was not analysed.')}")
    if detail:
        print(f"  {detail}")
    print("\n  No score, no phase scores and no coaching feedback are produced")
    print("  for a rejected video. A rejection is not a score of zero.")


def _bar(score: int) -> str:
    filled = int(round(BAR_WIDTH * score / 100.0))
    return "#" * filled + "." * (BAR_WIDTH - filled)


def _print_unsupported_shot(summary: ShotSummary) -> None:
    info = describe(summary.shot_type) if summary.shot_type else None
    label = info.label if info else "Unrecognised action"
    print(f"  SHOT {summary.shot_number}   NOT SCORED — {label}")
    print("=" * 72)
    reason = summary.rejection or RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET
    print(f"  {REJECTION_TEXT.get(reason, 'Not analysed.')}")
    if info and info.note:
        print(f"  {info.note}")
    if summary.classification:
        for note in summary.classification.evidence:
            print(f"    - {note}")
    print("\n  No score and no coaching feedback: scoring this with jump-shot")
    print("  rules would produce confident but wrong advice.")


def print_shot(summary: ShotSummary) -> None:
    print()
    print("=" * 72)
    if summary.is_rejected:
        _print_unsupported_shot(summary)
        return

    type_label = describe(summary.shot_type).label if summary.shot_type else "Shot"
    print(f"  SHOT {summary.shot_number}   {type_label}   "
          f"OVERALL {summary.score}/100  ({summary.grade})")
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


def print_session_summary(shots, discarded: int) -> None:
    scored = [s for s in shots if not s.is_rejected]
    unsupported = [s for s in shots if s.is_rejected]

    print()
    print("=" * 72)
    if scored:
        average = sum(s.score for s in scored) / len(scored)
        print(f"  SESSION: {len(scored)} scored shot(s), average {average:.0f}/100")
    else:
        print("  SESSION: no shots of a supported type were found.")
    if unsupported:
        print(f"  {len(unsupported)} shot(s) recognised but not yet supported "
              "— not scored.")
    if discarded:
        print(f"  {discarded} candidate movement(s) discarded as preparation, "
              "not shots.")
    print("=" * 72)


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

    # Phase 1 — validate metadata before anything temporal depends on it.
    meta = probe_video(video_path)
    if not meta.is_valid:
        print_rejection(RejectionReason.INVALID_VIDEO, meta.error or "")
        return 2
    if meta.warning:
        print(f"Note: {meta.warning}")

    profile = build_player_profile(height_cm=args.height_cm, shooting_hand=args.hand)

    cap = cv2.VideoCapture(str(video_path))
    fps = meta.fps

    detector = PoseDetector(vision.RunningMode.VIDEO)
    pipe = ShotAnalysisPipeline(enable_ball=False, player=profile)
    pipe.set_fps(fps)

    print(f"\nVideo        : {video_path.name}  ({fps:.2f} fps)")
    print(f"Player height: {profile.describe_height()}")
    if not profile.has_height:
        print("               -> height-relative metrics will be skipped")

    shots = []
    index = 0
    pose_frames = 0
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
        if result.has_pose:
            pose_frames += 1
        if result.shot_summary is not None:
            shots.append(result.shot_summary)
        index += 1
    cap.release()

    trailing = pipe.finalize_session()
    if trailing is not None:
        shots.append(trailing)

    pose_share = pose_frames / max(index, 1)

    # No person in frame => this is not footage of someone shooting.
    if pose_share < MIN_POSE_FRAME_SHARE:
        print_rejection(
            RejectionReason.NO_PERSON_DETECTED,
            f"A person was visible in only {pose_share * 100:.0f}% of frames "
            f"(needs at least {MIN_POSE_FRAME_SHARE * 100:.0f}%).",
        )
        return 3

    if not shots:
        print_rejection(
            RejectionReason.NO_SHOOTING_EVENT,
            f"{pipe.shot_tracker.discarded_candidates} candidate movement(s) were "
            "examined and none contained a release.",
        )
        return 4

    for summary in shots:
        print_shot(summary)

    print_session_summary(shots, pipe.shot_tracker.discarded_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
