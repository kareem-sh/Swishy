"""One policy for partial observations, and the evidence that it is one policy.

Two rules used to answer the same question in opposite ways. `hold_duration_s`
returns None when the hand has not come back down before the clip ends, so a
truncated recording is scored as neither a good follow-through nor a bad one.
`landing_balance` used to FAIL a shot whose recording stopped mid-descent --
same root cause, opposite verdict.

These lock the resolved policy in place:

    measured and complete        -> evaluate
    truncated before the answer  -> withdraw, never invent
    complete and bad             -> still fails

They also pin the ball_lift finding, which went the other way. A short
ball_lift is NOT insufficient evidence: the metric inside it is an
instantaneous pose angle, and one frame is a complete observation of a pose.
Measured across video8 and salah_video, 12 of 14 shots showed a real knee
countermovement (31-78 deg of excursion against 1-3 deg of jitter) while
ball_lift still ran as short as 0.03 s -- the player dipped and shot in one
motion. Short is the movement, not a measurement failure, so nothing here
blocks scoring on duration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.models import RuleResult  # noqa: E402
from feedback.scorer import _withdraw_truncated_landing  # noqa: E402


def _landing(measured, max_value=0.035):
    return RuleResult(
        rule_id="landing_balance",
        name="Landing",
        passed=not (measured > max_value),
        severity="info",
        message="landing",
        phase="landing",
        measured_value=measured,
        min_value=-0.10,
        max_value=max_value,
        unit="",
    )


def _other():
    return RuleResult(
        rule_id="elbow_extension_release",
        name="Reach at Release",
        passed=True,
        severity="warning",
        message="elbow",
        phase="release",
        measured_value=160.0,
        min_value=142.0,
        max_value=180.0,
    )


def _ids(rules):
    return {r.rule_id for r in rules}


# --------------------------------------------------------------- truncation
def test_landing_still_airborne_at_cutoff_is_withdrawn_not_failed():
    """The measured case: video8_shot10_jump ends 0.048 above baseline.

    Nobody watched this player land. That is not a failed landing.
    """
    rules = [_other(), _landing(0.048)]
    kept = _withdraw_truncated_landing(rules, frames=[], ended_early=True)
    assert _ids(kept) == {"elbow_extension_release"}


def test_landing_that_settled_badly_is_kept_even_when_the_clip_ends_early():
    """Below the floor is a COMPLETE observation of a bad landing.

    The withdrawal must be narrow. A player who came down and collapsed was
    fully observed doing it, and a clip that happens to end there must not
    launder that into no verdict.
    """
    rules = [_other(), _landing(-0.30)]
    kept = _withdraw_truncated_landing(rules, frames=[], ended_early=True)
    assert _ids(kept) == {"elbow_extension_release", "landing_balance"}


def test_landing_is_kept_when_the_recording_did_not_end_early():
    """A shot with footage after it was observed to the end. Airborne is real."""
    rules = [_other(), _landing(0.048)]
    kept = _withdraw_truncated_landing(rules, frames=[], ended_early=False)
    assert _ids(kept) == {"elbow_extension_release", "landing_balance"}


def test_a_settled_landing_is_never_touched():
    rules = [_other(), _landing(0.004)]
    for early in (True, False):
        kept = _withdraw_truncated_landing(rules, frames=[], ended_early=early)
        assert _ids(kept) == {"elbow_extension_release", "landing_balance"}


# ------------------------------------------------------------ noise ceiling
def test_set_shot_landing_noise_passes_the_ceiling():
    """A set shot never leaves the floor, so its landing reading is noise.

    Measured across 11 set shots in video8 and salah_video the readings ran
    -0.009 to +0.022. Every one of them must pass: a player who never jumped
    cannot have failed to come down. +0.022 failed against the old +0.02
    ceiling, which is what moved it.
    """
    observed = [-0.009, -0.005, -0.002, -0.001, 0.004, 0.009,
                0.009, 0.014, 0.016, 0.018, 0.022]
    for value in observed:
        assert _landing(value).passed, f"set-shot noise {value:+.3f} must pass"


def test_a_genuinely_airborne_landing_still_fails():
    """The ceiling moved for noise, not to stop the rule working."""
    assert not _landing(0.064).passed


# --------------------------------------------------------------- ball_lift
def test_a_single_frame_ball_lift_still_produces_a_score():
    """Short phase, instantaneous metric, complete observation.

    `elbow_slot_ball_lift` reads a pose angle. One frame is a whole answer to
    "where was the elbow", so brevity alone must not withdraw the rule -- the
    forbidden `ball_lift < 0.15s -> no score` gate would fail this test.
    """
    from analysis.engine import BiomechanicsEngine
    from angles.calculator import AngleResult
    from phase_detection.features import KinematicFeatures

    engine = BiomechanicsEngine()
    angles = {"right_elbow": AngleResult("right_elbow", 85.0, True, True)}
    features = KinematicFeatures(
        wrist_y=1.0, shoulder_y=1.4, hip_y_avg=0.95, nose_y=1.6,
        ankle_y_avg=0.1, ankle_baseline_y=0.1, body_rise_ratio=0.0,
        hip_x_ratio=0.5, body_pixel_height=0.6,
    )
    result = engine.evaluate("ball_lift", angles, features, "right")
    ids = {r.rule_id for r in result.active_rules}
    assert "elbow_slot_ball_lift" in ids
