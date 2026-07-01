"""Tests for Phase 5 shot scoring and feedback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import AnalysisResult, RuleResult
from angles.calculator import AngleResult
from feedback.generator import generate_coaching_tips
from feedback.scorer import score_shot
from feedback.shot_tracker import ShotTracker
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


def _snapshot(phase, rules):
    return FrameSnapshot(
        timestamp_ms=0,
        angles={"right_elbow": AngleResult("right_elbow", 170.0, True, True)},
        shooting_side="right",
        phase=phase,
        analysis=AnalysisResult(
            phase=phase,
            active_rules=rules,
            violations=[r for r in rules if not r.passed],
            passed_count=sum(1 for r in rules if r.passed),
            total_count=len(rules),
        ),
    )


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
    tracker = ShotTracker()
    snap = _snapshot("loading", [])
    assert tracker.update("loading", snap) is None
    assert tracker.shot_in_progress

    done = tracker.update("ready_stance", _snapshot("ready_stance", [_rule("a", True)]))
    assert done is None

    tracker._prev_phase = "landing"
    done = tracker.update("ready_stance", _snapshot("ready_stance", [_rule("a", True)]))
    assert done is not None
    assert done.shot_number == 1
    assert 0 <= done.score <= 100


def test_shot_tracker_starts_on_ball_lift():
    tracker = ShotTracker()
    tracker._prev_phase = "ready_stance"
    snap = _snapshot("ball_lift", [])
    assert tracker.update("ball_lift", snap) is None
    assert tracker.shot_in_progress


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
    test_coaching_tips_include_violations()
    print("All Phase 5 tests passed.")
