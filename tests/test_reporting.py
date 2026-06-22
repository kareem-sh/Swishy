"""Tests for detailed session reporting."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import AnalysisResult, RuleResult
from feedback.models import ShotSummary
from feedback.report_builder import build_detailed_shot_report, build_session_report
from feedback.report_models import KeyFrame
from feedback.report_writer import write_session_report


def _summary(passed=True):
    rule = RuleResult(
        rule_id="elbow_angle",
        name="Elbow Angle",
        passed=passed,
        severity="warning",
        message="Elbow too bent" if not passed else "Elbow OK",
        phase="release",
        measured_value=95.0 if not passed else 170.0,
        min_value=150.0,
        max_value=180.0,
    )
    return ShotSummary(
        shot_number=1,
        score=80 if passed else 55,
        passed_count=1 if passed else 0,
        total_count=1,
        passed_rules=[rule] if passed else [],
        violations=[] if passed else [rule],
        coaching_tips=["Keep elbow higher"],
        phases_seen=["loading", "release"],
    )


def _key_frame_pair(frame_index=10, rule_ids=None):
    kf = KeyFrame(
        frame_index=frame_index,
        timestamp_ms=500,
        timestamp_label="00:00.50",
        phase="release",
        phase_label="Release",
        image_filename="",
        capture_reason="Issue: Elbow Angle",
        issues=["Elbow too bent"],
        rule_ids=rule_ids or ["elbow_angle"],
        angles_summary="elbow=95°",
    )
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    return kf, img


def test_build_detailed_shot_report_links_violation_frame():
    summary = _summary(passed=False)
    detailed = build_detailed_shot_report(
        summary=summary,
        phase_moments=[{
            "phase": "release",
            "phase_label": "Release",
            "timestamp_ms": 500,
            "frame_index": 10,
        }],
        key_frame_pairs=[_key_frame_pair()],
        start_ms=0,
        end_ms=1000,
    )

    assert detailed.summary.shot_number == 1
    assert len(detailed.key_frames) == 1
    assert detailed.key_frames[0].image_filename.startswith("shot_01_frame_")
    assert detailed.frame_images[10] is not None

    failed = [ev for ev in detailed.rule_evaluations if not ev.passed]
    assert len(failed) == 1
    assert failed[0].key_frame_filename == detailed.key_frames[0].image_filename


def test_build_session_report_aggregates_improvements():
    shot_a = build_detailed_shot_report(
        summary=_summary(passed=False),
        phase_moments=[],
        key_frame_pairs=[_key_frame_pair()],
        start_ms=0,
        end_ms=1000,
    )
    shot_b = build_detailed_shot_report(
        summary=_summary(passed=True),
        phase_moments=[],
        key_frame_pairs=[],
        start_ms=1000,
        end_ms=2000,
    )

    session = build_session_report(
        session_id="test_session",
        source_type="video",
        source_name="test.mp4",
        fps=30.0,
        total_frames=60,
        shots=[shot_a, shot_b],
    )

    assert session.shot_count == 2
    assert session.overall_score is not None
    assert any("Elbow Angle" in item for item in session.top_improvements)


def test_write_session_report_creates_markdown_and_frames():
    detailed = build_detailed_shot_report(
        summary=_summary(passed=False),
        phase_moments=[{
            "phase": "release",
            "phase_label": "Release",
            "timestamp_ms": 500,
            "frame_index": 10,
        }],
        key_frame_pairs=[_key_frame_pair()],
        start_ms=0,
        end_ms=1000,
    )
    session = build_session_report(
        session_id="write_test",
        source_type="image",
        source_name="pose.jpg",
        fps=30.0,
        total_frames=1,
        shots=[detailed],
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = write_session_report(session, detailed.frame_images, Path(tmp))
        assert out.exists()
        assert out.name == "REPORT.md"
        content = out.read_text(encoding="utf-8")
        assert "Shooting Form Analysis Report" in content
        assert "Key Frames" in content
        frames_dir = out.parent / "frames"
        assert frames_dir.exists()
        assert any(frames_dir.glob("*.jpg"))


if __name__ == "__main__":
    test_build_detailed_shot_report_links_violation_frame()
    test_build_session_report_aggregates_improvements()
    test_write_session_report_creates_markdown_and_frames()
    print("All reporting tests passed.")
