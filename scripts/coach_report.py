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
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.quiet import quiet_native_stderr  # noqa: E402  — must precede mediapipe

from mediapipe.tasks.python import vision  # noqa: E402

from feedback.models import ShotSummary  # noqa: E402
from feedback.payload import session_to_dict  # noqa: E402
from pipeline import ShotAnalysisPipeline  # noqa: E402
from player.profile import PlayerProfile, build_player_profile  # noqa: E402
from pose.detector import PoseDetector  # noqa: E402
from shots.types import RejectionReason, describe  # noqa: E402
from utils.progress import ProgressReporter  # noqa: E402
from utils.video_meta import VideoMetadata, probe_video  # noqa: E402

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


# ---------------------------------------------------------------------------
# ANALYSIS — "what happened in this video?"
#
# Kept free of printing so the acceptance tests can drive the exact same
# production path the CLI drives. `main` below is presentation only: it calls
# `analyze_video` and formats the result. Nothing here is test-only.
# ---------------------------------------------------------------------------


@dataclass
class AnalysisRun:
    """Everything one video produced. The CLI prints it; tests assert on it."""

    video: Path
    fps: float
    shots: List[ShotSummary] = field(default_factory=list)
    pose_share: float = 0.0
    discarded_candidates: int = 0
    frames_read: int = 0
    rejection: Optional[RejectionReason] = None
    rejection_detail: str = ""

    @property
    def is_rejected(self) -> bool:
        """True when the whole video was rejected, not merely a single shot."""
        return self.rejection is not None

    @property
    def scored_shots(self) -> List[ShotSummary]:
        return [s for s in self.shots if not s.is_rejected]

    def to_payload(self) -> dict:
        """The JSON an API would return for this upload.

        One serialiser for the CLI's `--json` and for any future service, so
        the two can never drift into describing the same analysis differently.
        """
        return session_to_dict(
            video_name=self.video.name,
            fps=self.fps,
            frames_read=self.frames_read,
            shots=self.shots,
            discarded_candidates=self.discarded_candidates,
            rejection=self.rejection.value if self.rejection else None,
            rejection_detail=self.rejection_detail,
        )


def _decode_and_analyse(video_path, fps, pipe, progress):
    """Feed every frame through the pipeline.

    Every frame is read with a blocking read and none are skipped, and each
    frame's timestamp comes from its INDEX and the source frame rate -- never
    from the clock. Analysis is therefore identical whether the machine keeps
    up with real time or not; only the wait is longer.
    """
    shots: List[ShotSummary] = []
    index = 0
    pose_frames = 0

    with quiet_native_stderr():
        cap = cv2.VideoCapture(str(video_path))
        detector = PoseDetector(vision.RunningMode.VIDEO)
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
            index += 1
            if progress is not None:
                progress.advance(note="reading")
        cap.release()

    # Segmentation happens HERE, once, with the whole video decoded -- not
    # frame by frame on the way past.
    #
    # A shot is a shape in time: the hand rises out of a stance and comes back
    # down. Deciding at frame 400 whether frame 400 is part of one means
    # guessing at what frames 401-460 will do, and every timeout, hysteresis
    # window and latch in the old detector existed to cover that guess. None of
    # them is needed once the file has been read, because it has all already
    # happened.
    shots = pipe.segment_offline()
    if progress is not None:
        progress.advance(note=f"{len(shots)} shot(s) found")
    return shots, pose_frames, index


def analyze_video(
    video_path,
    height_cm: Optional[float] = None,
    shooting_hand: str = "auto",
    on_start: Optional[Callable[[VideoMetadata, PlayerProfile], None]] = None,
    progress_factory: Optional[Callable[[VideoMetadata], ProgressReporter]] = None,
    enable_ball: Optional[bool] = None,
) -> AnalysisRun:
    """Run one video through the real pipeline and return what was found.

    `on_start` and `progress_factory` are presentation hooks for the CLI.
    Neither affects analysis: every frame is read and processed either way.

    `enable_ball` decides whether the ball/rim detector runs.

        None   -> defer to config/ball.yaml  (the single source of truth)
        True   -> force on
        False  -> force off

    This is NOT only about reporting a make or a miss. With the detector
    active, `ShotTracker` waits for a ball outcome before closing a shot
    instead of closing on body motion alone, so the flag moves where one shot
    ends and the next begins. It used to be hardcoded to False here while
    `modes/video_mode.py` passed nothing and therefore read the config, which
    meant the same video could be segmented differently depending on which
    entry point you launched.
    """
    video_path = Path(video_path)

    # Validate metadata before anything temporal depends on it.
    meta = probe_video(video_path)
    if not meta.is_valid:
        return AnalysisRun(
            video=video_path,
            fps=0.0,
            rejection=RejectionReason.INVALID_VIDEO,
            rejection_detail=meta.error or "",
        )

    profile = build_player_profile(height_cm=height_cm, shooting_hand=shooting_hand)
    if on_start is not None:
        on_start(meta, profile)

    fps = meta.fps
    pipe = ShotAnalysisPipeline(enable_ball=enable_ball, player=profile)
    pipe.set_fps(fps)
    # Uploading a file is an offline task, so it is analysed as one.
    pipe.enable_offline_segmentation()

    progress = progress_factory(meta) if progress_factory is not None else None
    shots, pose_frames, index = _decode_and_analyse(video_path, fps, pipe, progress)
    if progress is not None:
        progress.finish(f"Analysed {index} frames in {progress.elapsed_s:.0f}s.")

    # No `finalize_session()` here. That exists to close an attempt the live
    # detector still had open when the video ended -- a problem that cannot
    # arise when segmentation runs after the last frame is already read.

    pose_share = pose_frames / max(index, 1)
    run = AnalysisRun(
        video=video_path,
        fps=fps,
        shots=shots,
        pose_share=pose_share,
        discarded_candidates=pipe.shot_tracker.discarded_candidates,
        frames_read=index,
    )

    # No person in frame => this is not footage of someone shooting.
    if pose_share < MIN_POSE_FRAME_SHARE:
        run.rejection = RejectionReason.NO_PERSON_DETECTED
        run.rejection_detail = (
            f"A person was visible in only {pose_share * 100:.0f}% of frames "
            f"(needs at least {MIN_POSE_FRAME_SHARE * 100:.0f}%)."
        )
    elif not shots:
        run.rejection = RejectionReason.NO_SHOOTING_EVENT
        run.rejection_detail = (
            f"{run.discarded_candidates} candidate movement(s) were examined "
            "and none contained a release."
        )
    return run


# ---------------------------------------------------------------------------
# PRESENTATION — "how do we tell the player?"
# ---------------------------------------------------------------------------


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


def _print_phase_notes(ph) -> None:
    """One phase's notes, or nothing at all if it has none.

    A phase whose only rules are the continuously-checked ones (posture, head
    stability) has nothing of its own to say -- those are gathered and
    reported once under "Through the Whole Shot". Printing an empty heading
    for such a phase just looks broken.
    """
    if not (ph.strengths or ph.refinements or ph.fixes or ph.measured):
        return
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


def _print_next_steps(summary: ShotSummary) -> None:
    if summary.next_rep_focus:
        print("\n  WORK ON THIS NEXT")
        for item in summary.next_rep_focus:
            print(f"    -> {item}")
    if summary.practice_drills:
        print("\n  DRILLS")
        for drill in summary.practice_drills:
            print(f"    * {drill}")


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
        _print_phase_notes(ph)
    _print_next_steps(summary)

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


EXIT_CODES = {
    RejectionReason.INVALID_VIDEO: 2,
    RejectionReason.NO_PERSON_DETECTED: 3,
    RejectionReason.NO_SHOOTING_EVENT: 4,
}


def _make_reporter(meta: VideoMetadata) -> ProgressReporter:
    return ProgressReporter(total_frames=meta.frame_count)


def _print_header(meta: VideoMetadata, profile: PlayerProfile) -> None:
    if meta.warning:
        print(f"Note: {meta.warning}")
    print(f"\n  Video        : {meta.path.name}")
    print(f"  Length       : {meta.duration_s:.1f}s, "
          f"{meta.frame_count} frames at {meta.fps:.0f} fps")
    print(f"  Player height: {profile.describe_height()}")
    if not profile.has_height:
        print("                 (height-relative metrics will be skipped)")
    print()


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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress bar (the report is unchanged)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the analysis as JSON instead of a text report "
             "(this is the payload an API would return)",
    )
    ball = parser.add_mutually_exclusive_group()
    ball.add_argument(
        "--ball",
        dest="enable_ball",
        action="store_true",
        default=None,
        help="force ball/rim detection on (slower: loads YOLO)",
    )
    ball.add_argument(
        "--no-ball",
        dest="enable_ball",
        action="store_false",
        help="force ball/rim detection off (pose only)",
    )
    args = parser.parse_args()

    show_progress = not args.quiet and not args.json

    run = analyze_video(
        args.video,
        height_cm=args.height_cm,
        shooting_hand=args.hand,
        on_start=None if args.json else _print_header,
        # The frame total is only known after probing, so the reporter is
        # built by analysis and owned by presentation.
        progress_factory=_make_reporter if show_progress else None,
        enable_ball=args.enable_ball,
    )

    if args.json:
        # Exactly what an API would return for this upload.
        print(json.dumps(run.to_payload(), indent=2, ensure_ascii=False))
        return 0 if not run.is_rejected else EXIT_CODES.get(run.rejection, 1)

    if run.is_rejected:
        print_rejection(run.rejection, run.rejection_detail)
        return EXIT_CODES.get(run.rejection, 1)

    for summary in run.shots:
        print_shot(summary)

    print_session_summary(run.shots, run.discarded_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
