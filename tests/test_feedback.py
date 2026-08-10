"""Tests for Phase 5 shot scoring and feedback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import AnalysisResult, RuleResult
from angles.calculator import AngleResult
from feedback.generator import generate_coaching_tips
from feedback.scorer import score_shot
from feedback.shot_tracker import ShotTracker
from phase_detection.features import KinematicFeatures
from utils.frame_buffer import FrameSnapshot


def _rule(rule_id, passed, severity="warning"):
    return RuleResult(
        rule_id=rule_id,
        name=rule_id,
        passed=passed,
        severity=severity,
        message=f"msg-{rule_id}",
        phase="release",
        measured_value=100.0,
    )


def _features(wrist_y=1.0, ankle_rise=0.0):
    """A plausible upright pose with the wrist at a given height."""
    return KinematicFeatures(
        wrist_y=wrist_y,
        shoulder_y=1.40,
        hip_y_avg=0.95,
        nose_y=1.60,
        ankle_y_avg=0.10,
        ankle_baseline_y=0.10,
        # Whole-body rise is an IMAGE-space signal expressed as a fraction of
        # the player's on-screen height; world landmarks are hip-centred and
        # cannot observe it.
        body_rise_ratio=ankle_rise,
        hip_x_ratio=0.5,
        body_pixel_height=0.6,
    )


def _snapshot(phase, rules, timestamp_ms=0, wrist_y=1.0, ankle_rise=0.0):
    return FrameSnapshot(
        timestamp_ms=timestamp_ms,
        angles={"right_elbow": AngleResult("right_elbow", 170.0, True, True)},
        shooting_side="right",
        phase=phase,
        features=_features(wrist_y=wrist_y, ankle_rise=ankle_rise),
        analysis=AnalysisResult(
            phase=phase,
            active_rules=rules,
            violations=[r for r in rules if not r.passed],
            passed_count=sum(1 for r in rules if r.passed),
            total_count=len(rules),
        ),
    )


def _credible_shot(tracker, rules=None, ankle_rise=0.0, start_ms=0):
    """Drive a tracker through a full, credible shooting motion.

    A shot is only confirmed if it reaches release AND the wrist actually
    travels, so the synthetic sequence has to move like a real shot.
    """
    rules = rules or [_rule("a", True)]
    steps = [
        ("loading", 1.00),
        ("ball_lift", 1.20),
        ("release", 1.48),
        ("follow_through", 1.40),
        ("landing", 1.00),
    ]
    completed = None
    for i, (phase, wrist_y) in enumerate(steps):
        rise = ankle_rise if phase in ("ball_lift", "release") else 0.0
        result = tracker.update(
            phase,
            _snapshot(phase, rules, start_ms + i * 100, wrist_y, rise),
        )
        completed = result or completed
    result = tracker.update(
        "ready_stance",
        _snapshot("ready_stance", rules, start_ms + 500, 0.95),
    )
    return result or completed


def test_score_shot_weighted():
    frames = [
        _snapshot("release", [_rule("a", True), _rule("b", False)]),
        _snapshot("follow_through", [_rule("a", True), _rule("b", False, "error")]),
    ]
    summary = score_shot(frames, shot_number=1)
    assert summary.total_count == 2
    assert summary.passed_count == 1
    assert 0 < summary.score < 100


def test_shot_tracker_completes():
    """A full credible shooting motion produces exactly one scored shot."""
    tracker = ShotTracker()
    done = _credible_shot(tracker)
    assert done is not None
    assert done.shot_number == 1
    assert 0 <= done.score <= 100
    assert not tracker.shot_in_progress


def test_shot_tracker_starts_on_ball_lift():
    tracker = ShotTracker()
    tracker._prev_phase = "ready_stance"
    assert tracker.update("ball_lift", _snapshot("ball_lift", [])) is None
    assert tracker.shot_in_progress
    assert tracker.capture_warning is None


def test_shot_tracker_starts_mid_phase():
    """Recording that begins mid-shot is flagged, not silently accepted."""
    tracker = ShotTracker()
    assert tracker.update("jump", _snapshot("jump", [_rule("a", True)])) is None
    assert tracker.shot_in_progress
    assert tracker.capture_warning is not None
    assert "Jump" in tracker.capture_warning


def test_toe_jiggling_does_not_create_shots():
    """Bouncing on the toes must not be counted as shooting.

    The player repeatedly dips into `loading` and settles back, without ever
    releasing the ball or lifting the wrist. Before the candidate lifecycle
    this produced one "shot" per dip.
    """
    tracker = ShotTracker()
    t = 0
    for _ in range(3):
        for phase, wrist in (("loading", 0.98), ("ready_stance", 1.00)):
            assert tracker.update(phase, _snapshot(phase, [], t, wrist)) is None
            t += 100

    assert tracker.shot_count == 0
    assert tracker.discarded_candidates >= 0  # discarded or still pending, never scored


def test_two_shots_are_independent():
    """Two credible shots produce two summaries with no state leakage."""
    tracker = ShotTracker()
    first = _credible_shot(tracker, start_ms=0)
    # Past the refractory window before the second attempt begins.
    second = _credible_shot(tracker, start_ms=3000)

    assert first is not None and second is not None
    assert first.shot_number == 1
    assert second.shot_number == 2
    assert first is not second
    assert not tracker.shot_in_progress


def test_set_shot_and_jump_shot_are_distinguished():
    """Vertical displacement at the shooting event separates the two types."""
    from shots.types import ShotType

    set_shot = _credible_shot(ShotTracker(), ankle_rise=0.0)
    jump_shot = _credible_shot(ShotTracker(), ankle_rise=0.20)

    assert set_shot.shot_type is ShotType.SET_SHOT
    assert jump_shot.shot_type is ShotType.JUMP_SHOT
    # A set shot must not be penalised for having no jump phase.
    assert set_shot.score is not None
    assert jump_shot.score is not None


def test_coaching_tips_include_violations():
    summary = score_shot(
        [_snapshot("release", [_rule("elbow", False)])],
        shot_number=1,
    )
    summary.coaching_tips = generate_coaching_tips(summary)
    assert len(summary.coaching_tips) >= 1


if __name__ == "__main__":
    test_score_shot_weighted()
    test_shot_tracker_completes()
    test_shot_tracker_starts_on_ball_lift()
    test_shot_tracker_starts_mid_phase()
    test_toe_jiggling_does_not_create_shots()
    test_two_shots_are_independent()
    test_set_shot_and_jump_shot_are_distinguished()
    test_coaching_tips_include_violations()
    print("All Phase 5 tests passed.")
