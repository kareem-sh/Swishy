"""Backward-compatible rule scoring and localization in the JSON payload."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import RuleOutcome, RuleResult
from feedback.models import PhaseScore, ShotSummary
from feedback.localization import phase_label_ar, rule_name_ar
from feedback.payload import _coaching_dict, _phase_dict, _rule_dict


def _good_rule() -> RuleResult:
    return RuleResult(
        rule_id="elbow_extension_release",
        name="Reach at Release",
        passed=True,
        severity="warning",
        message="Good reach. Extend a little further toward the rim to lift the arc.",
        phase="release",
        measured_value=150.0,
        min_value=142.0,
        max_value=180.0,
        ideal_min=160.0,
        ideal_max=175.0,
        scored=True,
        outcome=RuleOutcome.GOOD,
    )


def test_rule_json_keeps_message_and_adds_score_and_languages():
    rule = _good_rule()
    payload = _rule_dict(rule)

    # Existing clients keep reading the exact field they used before.
    assert payload["name"] == rule.name
    assert payload["message"] == rule.message
    # New clients can render the rule independently of the phase score.
    assert payload["name_localized"] == {
        "en": "Reach at Release",
        "ar": "امتداد الذراع عند الإطلاق",
    }
    assert payload["score"] == 72
    assert payload["feedback"]["en"] == rule.message
    assert payload["feedback"]["ar"]
    assert payload["feedback"]["ar"] != rule.message


def test_phase_json_contains_localized_rule_without_schema_replacement():
    phase = PhaseScore(
        phase="release",
        label="Release",
        score=72,
        rules=[_good_rule()],
    )
    payload = _phase_dict(phase)

    assert payload["phase"] == "release"
    assert payload["label"] == "Release"
    assert payload["label_localized"] == {
        "en": "Release",
        "ar": "الإطلاق",
    }
    assert payload["score"] == 72
    assert payload["rules"][0]["rule_id"] == "elbow_extension_release"
    assert payload["rules"][0]["name_localized"]["ar"] == (
        "امتداد الذراع عند الإطلاق"
    )
    assert set(payload["rules"][0]["feedback"]) == {"en", "ar"}


def test_measured_only_rule_has_no_fake_score():
    rule = RuleResult(
        rule_id="head_stability",
        name="Eyes and Head",
        passed=True,
        severity="info",
        message="Head movement, for reference — this one isn't scored.",
        phase="release",
        measured_value=0.06,
        unit="m/s",
        scored=False,
        outcome=RuleOutcome.EXCELLENT,
    )
    payload = _phase_dict(
        PhaseScore(
            phase="release",
            label="Release",
            score=None,
            measured=[rule],
        )
    )["measured_only"][0]

    assert payload["score"] is None
    assert payload["name_localized"] == {
        "en": "Eyes and Head",
        "ar": "ثبات العينين والرأس",
    }
    assert "لا تدخل في النتيجة" in payload["feedback"]["ar"]


def test_trajectory_and_whole_shot_phase_names_are_localized():
    trajectory_rule = RuleResult(
        rule_id="trajectory_release_angle",
        name="Ball Release Angle",
        passed=True,
        severity="warning",
        message="The ball left at the target launch angle.",
        phase="trajectory",
        measured_value=-2.28,
        unit="deg",
        scored=True,
        outcome=RuleOutcome.EXCELLENT,
    )
    trajectory = _phase_dict(
        PhaseScore(
            phase="trajectory",
            label="Trajectory",
            score=96,
            rules=[trajectory_rule],
        )
    )
    whole_shot = _phase_dict(
        PhaseScore(
            phase="whole_shot",
            label="Through the Whole Shot",
            score=None,
        )
    )

    assert trajectory["label_localized"]["ar"] == "مسار الكرة"
    assert trajectory["rules"][0]["name_localized"]["ar"] == (
        "زاوية إطلاق الكرة"
    )
    assert whole_shot["label_localized"]["ar"] == "طوال التسديدة"


def test_unknown_display_names_fall_back_to_english():
    assert rule_name_ar("future_rule", "Future Rule") == "Future Rule"
    assert phase_label_ar("future_phase", "Future Phase") == "Future Phase"


@pytest.mark.parametrize("section", ["rules", "trajectory_rules"])
def test_every_configured_rule_has_an_arabic_name(section):
    config_path = Path(__file__).resolve().parent.parent / "config" / "biomechanics.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    for rule_id, rule_config in config[section].items():
        english_name = rule_config["name"]
        assert rule_name_ar(rule_id, english_name) != english_name


def test_coaching_keeps_legacy_arrays_and_adds_bilingual_sections():
    rule = _good_rule()
    drill = (
        "Extension pause: at the top of each shot, freeze with full elbow "
        "extension for 1 second before follow-through — 15 reps."
    )
    summary = ShotSummary(
        shot_number=1,
        score=72,
        passed_count=1,
        total_count=1,
        coaching_tips=[rule.message, "Fair attempt — focus on one correction per rep."],
        next_rep_focus=[rule.message],
        practice_drills=[drill],
        phase_scores=[
            PhaseScore(
                phase="release",
                label="Release",
                score=72,
                rules=[rule],
            )
        ],
    )
    payload = _coaching_dict(summary)

    assert payload["summary"] == summary.coaching_tips
    assert payload["focus_next_rep"] == summary.next_rep_focus
    assert payload["drills"] == summary.practice_drills
    assert payload["localized"]["summary"]["en"] == summary.coaching_tips
    assert payload["localized"]["summary"]["ar"][0] != rule.message
    assert payload["localized"]["focus_next_rep"]["ar"][0] != rule.message
    assert payload["localized"]["drills"]["ar"][0] != drill
