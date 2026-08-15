"""Print and persist shot feedback when a rep completes."""

import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT
from feedback.models import ShotSummary
from feedback.payload import shot_to_dict


def print_shot_summary(summary: ShotSummary):
    score_text = f"{summary.score}/100" if summary.score is not None else "NOT SCORED"
    print("\n" + "=" * 50)
    print(f"  SHOT #{summary.shot_number}  —  {summary.grade.upper()}  ({score_text})")
    print("=" * 50)
    print(f"  Rules: {summary.passed_count}/{summary.total_count} passed")

    if summary.outcome is not None:
        result = summary.outcome.result.upper()
        confidence = summary.outcome.confidence
        print(f"  Basket: {result} ({confidence:.0%} confidence)")

    if (
        summary.pose_release_timestamp_ms is not None
        or summary.ball_release_timestamp_ms is not None
    ):
        print(
            "  Release timing: "
            f"pose={summary.pose_release_timestamp_ms} ms, "
            f"ball={summary.ball_release_timestamp_ms} ms, "
            f"difference={summary.release_disagreement_ms} ms, "
            f"alignment confidence={summary.release_alignment_confidence}"
        )

    if summary.capture_note:
        print(f"\n  Capture: {summary.capture_note}")

    if summary.next_rep_focus:
        print("\n  Next rep focus:")
        for item in summary.next_rep_focus:
            print(f"    * {item}")

    if summary.passed_rules:
        print("\n  Passed:")
        for rule in summary.passed_rules:
            print(f"    + {rule.name}")

    if summary.violations:
        print("\n  Fix next:")
        for rule in summary.violations:
            print(f"    - {rule.name}: {rule.message}")

    if summary.practice_drills:
        print("\n  Drills:")
        for drill in summary.practice_drills:
            print(f"    > {drill}")

    if summary.coaching_tips:
        print("\n  Coach says:")
        for tip in summary.coaching_tips:
            print(f"    > {tip}")

    print("=" * 50 + "\n")

    # json_path = save_shot_summary_json(summary)
    # print(f"  Shot JSON saved: {json_path}\n")


def shot_summary_to_dict(
    summary: ShotSummary,
    start_timestamp_ms: int | None = None,
    end_timestamp_ms: int | None = None,
) -> dict:
    """Convert one shot summary into a JSON-compatible dictionary.

    Delegates to `feedback.payload`, which reports only what was measured.
    This wrapper used to accept `shot_type="jump_shot"` and
    `court_location="right_wing_three_point_line"` as defaults and emit them
    verbatim, so every shot was described as a jump shot from the right wing
    no matter what the classifier had actually decided.
    """
    return shot_to_dict(summary, start_timestamp_ms, end_timestamp_ms)


def save_shot_summary_json(
    summary: ShotSummary,
    output_dir: str | Path | None = None,
) -> dict:
    """Save one shot as JSON and return the same JSON-compatible dictionary."""
    payload = shot_summary_to_dict(summary)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else PROJECT_ROOT / "outputs" / "shot_feedback"
    )
    destination.mkdir(parents=True, exist_ok=True)

    saved_at = datetime.now()
    timestamp = saved_at.strftime("%Y%m%d_%H%M%S_%f")
    output_path = destination / (
        f"shot_{summary.shot_number:02d}_{timestamp}.json"
    )
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def detailed_shot_to_dict(detailed_shot) -> dict:
    """Convert a DetailedShotReport into a JSON-compatible shot dictionary.

    A DetailedShotReport knows when the attempt began and ended, so these are
    real timestamps rather than the 1000/2500 ms placeholders they replaced.
    """
    return shot_to_dict(
        detailed_shot.summary,
        detailed_shot.start_timestamp_ms,
        detailed_shot.end_timestamp_ms,
    )


def session_report_to_dict(report) -> dict:
    """Convert a completed video/session report, including every shot, to JSON data."""
    shots = [detailed_shot_to_dict(shot) for shot in report.shots]

    return {
        "session_id": report.session_id,
        "source_type": report.source_type,
        "source_name": report.source_name,
        "fps": report.fps,
        "total_frames": report.total_frames,
        "shot_count": len(shots),
        "overall_score": report.overall_score,
        "overall_grade": report.overall_grade,
        "top_improvements": list(report.top_improvements),
        "strengths": list(report.strengths),
        "practice_plan": list(report.practice_plan),
        "session_notes": list(report.session_notes),
        "shots": shots,
    }


def save_session_report_json(report, output_path: str | Path) -> Path:
    """Persist a full session report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(session_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
