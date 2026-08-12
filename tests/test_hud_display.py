"""Tests for HUD display smoothing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angles.calculator import AngleResult
from pipeline import FrameResult
from visualization.hud_display import HudDisplaySmoother


def _frame(phase="release", phase_label="Release", elbow=170.0, violations=None):
    rules = violations or []
    from analysis.models import AnalysisResult, RuleResult

    active = [
        RuleResult(
            rule_id=f"r{i}",
            name=f"rule{i}",
            passed=False,
            severity="warning",
            message=msg,
            phase=phase,
        )
        for i, msg in enumerate(rules)
    ]
    return FrameResult(
        has_pose=True,
        phase=phase,
        phase_label=phase_label,
        shooting_side="right",
        angles={
            "right_elbow": AngleResult("right_elbow", elbow, True, True),
            "right_knee": AngleResult("right_knee", 120.0, True, True),
            "trunk": AngleResult("trunk", 15.0, True, True),
        },
        analysis=AnalysisResult(
            phase=phase,
            active_rules=active,
            violations=active,
            passed_count=0,
            total_count=len(active) or 1,
        ),
        shot_in_progress=True,
    )


def test_phase_label_requires_stability():
    smoother = HudDisplaySmoother()
    smoother._phase_stable_frames = 3
    smoother._phase_min_hold_frames = 5

    for _ in range(2):
        hud = smoother.update(_frame(phase="loading", phase_label="Loading"))
    assert hud.phase_label == "Ready Stance"

    for _ in range(3):
        hud = smoother.update(_frame(phase="loading", phase_label="Loading"))
    assert hud.phase_label == "Loading"


def test_angle_display_holds_between_small_changes():
    smoother = HudDisplaySmoother()
    smoother._angle_alpha = 0.5
    smoother._angle_step = 5
    smoother._angle_min_hold = 8

    hud = smoother.update(_frame(elbow=170.0))
    first = hud.angles["elbow"].text

    for deg in (171, 172, 173, 174):
        hud = smoother.update(_frame(elbow=deg))

    assert hud.angles["elbow"].text == first


def test_violations_hold_between_updates():
    smoother = HudDisplaySmoother()
    smoother._violation_hold = 10

    hud1 = smoother.update(_frame(violations=["Bend knees more"]))
    assert len(hud1.violations) == 1
    assert "Bend knees" in hud1.violations[0].message

    hud2 = smoother.update(_frame(violations=["Different message"]))
    assert "Bend knees" in hud2.violations[0].message


def test_invalid_angle_holds_last_valid_text():
    smoother = HudDisplaySmoother()
    smoother._angle_invalid_hold = 5

    smoother.update(_frame(elbow=170.0))
    invalid = AngleResult("right_elbow", None, False, False)
    hud = smoother.update(
        FrameResult(
            has_pose=True,
            phase="release",
            phase_label="Release",
            shooting_side="right",
            angles={"right_elbow": invalid, "right_knee": AngleResult("right_knee", 120.0, True, True), "trunk": AngleResult("trunk", 15.0, True, True)},
            shot_in_progress=True,
        )
    )
    assert hud.angles["elbow"].text.startswith("~")
    assert hud.angles["elbow"].text != "N/A"


if __name__ == "__main__":
    test_phase_label_requires_stability()
    test_angle_display_holds_between_small_changes()
    test_violations_hold_between_updates()
    test_invalid_angle_holds_last_valid_text()
    print("All HUD display tests passed.")
