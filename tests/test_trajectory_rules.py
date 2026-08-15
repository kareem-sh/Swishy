"""Trajectory-rule evaluation and shot-score integration."""

import contextlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.engine import BiomechanicsEngine
from analysis.models import AnalysisResult, RuleResult
from angles.calculator import AngleResult
from ball.models import ShotOutcome
from feedback.scorer import score_shot
from feedback.shot_tracker import ShotTracker
from phase_detection.features import KinematicFeatures
from phase_detection.phases import CORE_REST
from utils.frame_buffer import FrameSnapshot


def _comparison(**overrides) -> dict:
    values = {
        "release_angle_error_deg": 0.0,
        "release_speed_error_m_s": 0.0,
        "apex_height_error_rim_radii": 0.0,
        "apex_position_error_progress": 0.0,
        "apex_position_error_m": 0.0,
        "apex_fit_status": "accepted",
        "observed_kinematics_source": "early_post_release_fit",
        "observed_direct_point_count": 8,
        "observed_point_count": 40,
    }
    values.update(overrides)
    return values


def _pose_frame(rule: RuleResult) -> FrameSnapshot:
    return FrameSnapshot(
        timestamp_ms=1000,
        angles={},
        shooting_side="right",
        phase="release",
        analysis=AnalysisResult(
            phase="release",
            active_rules=[rule],
            violations=[] if rule.passed else [rule],
            passed_count=1 if rule.passed else 0,
            total_count=1,
        ),
    )


def _motion_frame(phase: str, timestamp_ms: int, wrist_y: float) -> FrameSnapshot:
    features = KinematicFeatures(
        wrist_y=wrist_y,
        shoulder_y=1.40,
        hip_y_avg=0.95,
        nose_y=1.60,
        ankle_y_avg=0.10,
        ankle_baseline_y=0.10,
        body_rise_ratio=0.0,
        hip_x_ratio=0.5,
        body_pixel_height=0.6,
    )
    return FrameSnapshot(
        timestamp_ms=timestamp_ms,
        angles={
            "right_elbow": AngleResult(
                "right_elbow", 160.0, True, True
            )
        },
        shooting_side="right",
        phase=phase,
        features=features,
        analysis=AnalysisResult(phase=phase),
    )


def test_trajectory_rules_apply_to_set_and_jump_shots() -> None:
    engine = BiomechanicsEngine()

    for shot_type in ("set_shot", "jump_shot"):
        analysis = engine.evaluate_trajectory(
            _comparison(),
            shot_type=shot_type,
        )
        assert analysis.phase == "trajectory"
        assert len(analysis.active_rules) == 5
        assert len([r for r in analysis.active_rules if r.scored]) == 4
        assert analysis.passed_count == 5
        assert analysis.violations == []


def test_trajectory_rules_produce_directional_feedback() -> None:
    analysis = BiomechanicsEngine().evaluate_trajectory(
        _comparison(
            release_angle_error_deg=-9.0,
            release_speed_error_m_s=1.1,
            apex_height_error_rim_radii=1.6,
            apex_position_error_progress=-0.16,
            apex_position_error_m=-0.8,
        ),
        shot_type="set_shot",
    )
    by_id = {rule.rule_id: rule for rule in analysis.active_rules}

    assert not by_id["trajectory_release_angle"].passed
    assert "too flat" in by_id["trajectory_release_angle"].message
    assert not by_id["trajectory_release_speed"].passed
    assert "too fast" in by_id["trajectory_release_speed"].message
    assert not by_id["trajectory_apex_height"].passed
    assert "too high" in by_id["trajectory_apex_height"].message
    assert not by_id["trajectory_apex_progress"].passed
    assert "too early" in by_id["trajectory_apex_progress"].message
    assert not by_id["trajectory_apex_distance"].scored


def test_failed_apex_fit_skips_apex_rules_without_inventing_zero() -> None:
    analysis = BiomechanicsEngine().evaluate_trajectory(
        _comparison(
            apex_fit_status="insufficient_points",
            apex_height_error_rim_radii=None,
            apex_position_error_progress=None,
            apex_position_error_m=None,
        ),
        shot_type="jump_shot",
    )

    assert {
        rule.rule_id for rule in analysis.active_rules
    } == {
        "trajectory_release_angle",
        "trajectory_release_speed",
    }


def test_trajectory_rules_join_pose_score_and_json_phase_structure() -> None:
    pose_rule = RuleResult(
        rule_id="synthetic_pose_rule",
        name="Synthetic Pose Rule",
        passed=True,
        severity="warning",
        message="Pose is on target.",
        phase="release",
        measured_value=0.0,
    )
    trajectory = BiomechanicsEngine().evaluate_trajectory(
        _comparison(),
        shot_type="set_shot",
    )
    summary = score_shot(
        [_pose_frame(pose_rule)],
        shot_number=1,
        shot_type="set_shot",
        shot_level_rules=trajectory.active_rules,
    )

    assert summary.total_count == 5  # one pose + four scored trajectory rules
    assert summary.passed_count == 5
    trajectory_phase = next(
        phase for phase in summary.phase_scores if phase.phase == "trajectory"
    )
    assert len(trajectory_phase.rules) == 4
    assert len(trajectory_phase.measured) == 1
    assert trajectory_phase.measured[0].rule_id == "trajectory_apex_distance"


def test_shot_tracker_attaches_trajectory_rules_to_completed_set_shot() -> None:
    tracker = ShotTracker()
    tracker.attach_analyzer(BiomechanicsEngine())
    tracker.configure_ball_outcome(required=True, body_grace_ms=500)

    steps = [
        ("rise", 0, 1.00),
        ("rise", 100, 1.20),
        ("release", 200, 1.48),
        ("recovery", 300, 1.40),
        ("recovery", 400, 1.00),
    ]
    for phase, timestamp_ms, wrist_y in steps:
        assert tracker.update(
            phase,
            _motion_frame(phase, timestamp_ms, wrist_y),
        ) is None

    outcome = ShotOutcome(
        result="made",
        confidence=0.95,
        release_timestamp_ms=200,
        outcome_timestamp_ms=500,
        timeseries_summary={"trajectory_comparison": _comparison()},
    )
    summary = tracker.update(
        CORE_REST,
        _motion_frame(CORE_REST, 500, 0.95),
        ball_outcome=outcome,
    )

    assert summary is not None
    assert summary.outcome is outcome
    trajectory_phase = next(
        phase for phase in summary.phase_scores if phase.phase == "trajectory"
    )
    assert len(trajectory_phase.rules) == 4
    assert len(trajectory_phase.measured) == 1


def test_offline_window_keeps_ball_trajectory_rules() -> None:
    tracker = ShotTracker()
    tracker.attach_analyzer(BiomechanicsEngine())
    frames = [
        _motion_frame("rise", 0, 1.00),
        _motion_frame("rise", 100, 1.20),
        _motion_frame("release", 200, 1.48),
        _motion_frame("recovery", 300, 1.40),
        _motion_frame("recovery", 400, 1.00),
    ]
    outcome = ShotOutcome(
        result="made",
        confidence=0.95,
        release_timestamp_ms=220,
        outcome_timestamp_ms=500,
        timeseries_summary={"trajectory_comparison": _comparison()},
    )

    summary = tracker.finalize_window(
        frames,
        peak_ms=200,
        ball_outcome=outcome,
    )

    assert summary is not None
    assert summary.outcome is outcome
    trajectory_phase = next(
        phase for phase in summary.phase_scores if phase.phase == "trajectory"
    )
    assert len(trajectory_phase.rules) == 4
    assert len(trajectory_phase.measured) == 1


def test_console_prints_trajectory_separately_from_pose_phases() -> None:
    from scripts.coach_report import print_shot

    trajectory = BiomechanicsEngine().evaluate_trajectory(
        _comparison(release_angle_error_deg=4.0),
        shot_type="set_shot",
    )
    summary = score_shot(
        [],
        shot_number=1,
        shot_type="set_shot",
        shot_level_rules=trajectory.active_rules,
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_shot(summary)
    rendered = output.getvalue()

    assert "POSE PHASE SCORES" in rendered
    assert "BALL TRAJECTORY" in rendered
    assert "not as a pose phase" in rendered
    assert "Ball Release Angle: +4.00 deg error" in rendered


if __name__ == "__main__":
    test_trajectory_rules_apply_to_set_and_jump_shots()
    test_trajectory_rules_produce_directional_feedback()
    test_failed_apex_fit_skips_apex_rules_without_inventing_zero()
    test_trajectory_rules_join_pose_score_and_json_phase_structure()
    test_shot_tracker_attaches_trajectory_rules_to_completed_set_shot()
    test_offline_window_keeps_ball_trajectory_rules()
    test_console_prints_trajectory_separately_from_pose_phases()
    print("All trajectory rule tests passed.")
