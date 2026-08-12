"""Tests for performance plan and mid-capture reporting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import RuleResult
from feedback.models import ShotSummary
from feedback.performance_plan import build_shot_performance_plan
from feedback.report_builder import build_detailed_shot_report, build_session_report


def _violation(rule_id="elbow_extension_release"):
    return RuleResult(
        rule_id=rule_id,
        name="Elbow Extension",
        passed=False,
        severity="warning",
        message="Extend elbow fully at release",
        phase="release",
        measured_value=140.0,
    )


def test_mid_start_capture_note():
    summary = ShotSummary(
        shot_number=1,
        score=70,
        passed_count=1,
        total_count=2,
        violations=[_violation()],
        phases_seen=["jump", "release", "follow_through", "landing"],
        started_mid_phase=True,
        entry_phase="jump",
    )
    summary = build_shot_performance_plan(summary)
    assert summary.started_mid_phase
    assert "mid-shot" in summary.capture_note.lower()
    assert "Loading" in summary.capture_note or "loading" in summary.missing_phases or len(summary.missing_phases) > 0
    assert len(summary.practice_drills) >= 1
    assert len(summary.next_rep_focus) >= 1


def test_session_report_includes_practice_plan():
    summary = build_shot_performance_plan(
        ShotSummary(
            shot_number=1,
            score=60,
            passed_count=0,
            total_count=1,
            violations=[_violation()],
            phases_seen=["release"],
            started_mid_phase=True,
            entry_phase="release",
        )
    )
    detailed = build_detailed_shot_report(
        summary=summary,
        phase_moments=[],
        key_frame_pairs=[],
        start_ms=0,
        end_ms=1000,
    )
    session = build_session_report(
        session_id="t",
        source_type="video",
        source_name="clip.mp4",
        fps=30.0,
        total_frames=30,
        shots=[detailed],
    )
    assert session.practice_plan
    assert session.session_notes
    assert any("mid-rep" in n for n in session.session_notes)


if __name__ == "__main__":
    test_mid_start_capture_note()
    test_session_report_includes_practice_plan()
    print("Performance plan tests passed.")
