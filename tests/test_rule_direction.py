"""A band must not contradict the direction its own metric runs in.

WHY THIS TEST EXISTS
--------------------
Bands are derived from a reference group by a single formula:

    acceptable = group range  +/- measurement uncertainty
    perfect    = group IQR,   widened to the uncertainty

That formula assumes deviation in EITHER direction is a fault, which is true of
a knee angle and false of half the rules here. Applied blindly when the pose
model changed, it proposed:

    trunk_posture     min: 4      -- would FAIL a perfectly upright trunk
    release_height    max: 0.213  -- would score a release HIGHER than the
                                     reference as worse than the reference
    knee_excursion    min: 1      -- would accept a shot with no
                                     countermovement, the exact thing the rule
                                     exists to catch
    landing_balance   max: 0.02   -- inside the metric's own measured noise

All four were caught by reading them. That is not a control. Each rule now
DECLARES its direction and this test enforces the declaration, so the next
re-derivation -- against a new model, a new coach, or a wider reference group --
cannot quietly reintroduce the same class of error.

The rules are the specification; this file only checks they agree with
themselves.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG = Path(__file__).resolve().parent.parent / "config" / "biomechanics.yaml"
RULES = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["rules"]
IDS = sorted(RULES)

VALID = {"two_sided", "higher_better", "lower_better"}


def bands(rule, model=None):
    """The band a given model would see, after `calibration:` is applied."""
    out = {k: rule.get(k) for k in ("min", "max", "ideal_min", "ideal_max")}
    if model:
        for k, v in ((rule.get("calibration") or {}).get(model) or {}).items():
            out[k] = v
    return out


def models_for(rule):
    return [None, *sorted((rule.get("calibration") or {}))]


@pytest.mark.parametrize("rule_id", IDS)
def test_every_rule_declares_a_direction(rule_id):
    """An undeclared direction is how the formula gets applied blindly."""
    d = RULES[rule_id].get("direction")
    assert d in VALID, f"{rule_id}: direction is {d!r}, expected one of {VALID}"


@pytest.mark.parametrize("rule_id", IDS)
def test_lower_better_rules_never_fail_a_perfect_value(rule_id):
    """`lower_better` means zero is the target. A floor above it fails success.

    `trunk_posture` and `landing_balance` both measure distance from an ideal
    of zero -- an upright trunk, a body back at its own baseline. A `min` above
    zero would report the best achievable value as a fault.
    """
    rule = RULES[rule_id]
    if rule.get("direction") != "lower_better":
        pytest.skip("not a lower-better metric")
    for model in models_for(rule):
        b = bands(rule, model)
        where = f"{rule_id}[{model or 'default'}]"
        if b["min"] is not None:
            assert b["min"] <= 0, f"{where}: min {b['min']} would fail a perfect 0"
        if b["ideal_min"] is not None:
            assert b["ideal_min"] <= 0, (
                f"{where}: ideal_min {b['ideal_min']} excludes a perfect 0")


@pytest.mark.parametrize("rule_id", IDS)
def test_higher_better_rules_do_not_cap_the_ideal_band(rule_id):
    """`higher_better` means more is never worse. An ideal ceiling says it is.

    An `ideal_max` on such a rule scores a player who EXCEEDS the reference as
    worse than the reference, which is the one thing a reference must never do.
    An outer `max` is still allowed: that is a gross-deviation bound, not a
    statement that more is worse.
    """
    rule = RULES[rule_id]
    if rule.get("direction") != "higher_better":
        pytest.skip("not a higher-better metric")
    for model in models_for(rule):
        b = bands(rule, model)
        where = f"{rule_id}[{model or 'default'}]"
        if b["ideal_max"] is None:
            continue
        # Only defensible when it sits at the metric's physical limit, where it
        # cannot penalise anyone -- an elbow cannot extend past 180 degrees.
        assert b["max"] is not None and b["ideal_max"] >= b["max"], (
            f"{where}: ideal_max {b['ideal_max']} caps improvement below the "
            f"outer max {b['max']}, so beating the reference costs points")


@pytest.mark.parametrize("rule_id", IDS)
def test_bands_are_ordered_and_nested(rule_id):
    """ideal must sit inside acceptable, and both must run low to high.

    Cheap, and it catches a whole class of copy-paste error in the per-model
    calibration blocks that would otherwise only show up as a strange score.
    """
    rule = RULES[rule_id]
    for model in models_for(rule):
        b = bands(rule, model)
        where = f"{rule_id}[{model or 'default'}]"
        if b["min"] is not None and b["max"] is not None:
            assert b["min"] < b["max"], f"{where}: min >= max"
        if b["ideal_min"] is not None and b["ideal_max"] is not None:
            assert b["ideal_min"] < b["ideal_max"], f"{where}: ideal inverted"
        if b["min"] is not None and b["ideal_min"] is not None:
            assert b["ideal_min"] >= b["min"], f"{where}: ideal_min below min"
        if b["max"] is not None and b["ideal_max"] is not None:
            assert b["ideal_max"] <= b["max"], f"{where}: ideal_max above max"


@pytest.mark.parametrize("rule_id", IDS)
def test_every_calibration_block_names_a_real_model(rule_id):
    """A typo in a model key is silent: the block is simply never selected."""
    known = {"lite", "full", "heavy"}
    for model in (RULES[rule_id].get("calibration") or {}):
        assert model in known, f"{rule_id}: unknown calibration model {model!r}"
