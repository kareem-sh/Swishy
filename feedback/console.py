"""Print and persist shot feedback when a rep completes."""

import json
from datetime import datetime
from pathlib import Path

from config.settings import PROJECT_ROOT
from feedback.models import ShotSummary


def print_shot_summary(summary: ShotSummary):
    print("\n" + "=" * 50)
    print(f"  SHOT #{summary.shot_number}  —  {summary.grade.upper()}  ({summary.score}/100)")
    print("=" * 50)
    print(f"  Rules: {summary.passed_count}/{summary.total_count} passed")

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
    shot_start_time_ms: int = 1000,
    shot_end_time_ms: int = 2500,
    shot_type: str = "jump_shot",
    court_location: str = "right_wing_three_point_line",
) -> dict:
    """Convert one shot summary into a JSON-compatible dictionary."""
    saved_at = datetime.now()
    return {
        "shot_number": summary.shot_number,
        "grade": summary.grade,
        "score": summary.score,
        "rules": {
            "passed_count": summary.passed_count,
            "total_count": summary.total_count,
        },
        "capture_note": summary.capture_note,
        "next_rep_focus": list(summary.next_rep_focus),
        "passed_rules": [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "message": rule.message,
            }
            for rule in summary.passed_rules
        ],
        "violations": [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "severity": rule.severity,
                "message": rule.message,
            }
            for rule in summary.violations
        ],
        "practice_drills": list(summary.practice_drills),
        "coaching_tips": list(summary.coaching_tips),
        "shot_metadata": {
            "start_time_ms": shot_start_time_ms,
            "end_time_ms": shot_end_time_ms,
            "shot_type": shot_type,
            "court_location": court_location,
            "is_placeholder": True,
        },
        "saved_at": saved_at.isoformat(timespec="milliseconds"),
    }


def save_shot_summary_json(
    summary: ShotSummary,
    output_dir: str | Path | None = None,
    shot_start_time_ms: int = 1000,
    shot_end_time_ms: int = 2500,
    shot_type: str = "jump_shot",
    court_location: str = "right_wing_three_point_line",
) -> dict:
    """Save one shot as JSON and return the same JSON-compatible dictionary."""
    payload = shot_summary_to_dict(
        summary=summary,
        shot_start_time_ms=shot_start_time_ms,
        shot_end_time_ms=shot_end_time_ms,
        shot_type=shot_type,
        court_location=court_location,
    )
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


def session_report_to_dict(report) -> dict:
    """Convert a completed video/session report, including every shot, to JSON data."""
    shots = [
        shot_summary_to_dict(
            summary=shot.summary,
        )
        for shot in report.shots
    ]

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
