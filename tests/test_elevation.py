"""Pin the measured behaviour of takeoff elevation.

The numbers asserted here were measured on labelled footage -- video8, whose
per-shot windows are frame-exact in single_shot/manifest.json, and salah_video,
labelled by the user. They are recorded so that a change which quietly breaks
shot typing fails here instead of on a thesis demo.

Reproduce with:  python scripts/peak_probe.py peaks --prominence 0.55
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shots.classifier import (  # noqa: E402
    JUMP_VERTICAL_DISPLACEMENT_RATIO,
    UNMEASURED_CONFIDENCE_CEILING,
    AttemptEvidence,
    classify,
)
from shots.elevation import (  # noqa: E402
    MIN_BASELINE_SAMPLES,
    shooting_event_ms,
    takeoff_elevation,
)
from shots.types import ShotType  # noqa: E402


def _clip(stance_ankle, event_ankle, fps=30.0, body_height=0.32):
    """Build one attempt: 2 s of standing, then a shooting event at t=0.

    Timestamps are negative before the event so the event sits at 0 ms, which
    is what the production caller passes after locating the wrist peak.
    """
    timestamps, ankles, heights, wrists = [], [], [], []
    step = int(1000 / fps)
    for ts in range(-2500, 700, step):
        timestamps.append(ts)
        heights.append(body_height)
        if -500 <= ts <= 500:
            ankles.append(event_ankle)
            # A peak in the middle of the event window.
            wrists.append(0.7 - abs(ts) / 5000.0)
        else:
            ankles.append(stance_ankle)
            wrists.append(-0.05)
    return timestamps, ankles, heights, wrists


def test_feet_on_the_floor_measure_no_elevation():
    ts, ankles, heights, wrists = _clip(stance_ankle=0.94, event_ankle=0.94)
    assert takeoff_elevation(ts, ankles, heights, wrists, 0) == 0.0


def test_elevation_is_expressed_in_body_heights():
    # Feet 0.032 of the frame higher, with a body 0.32 of the frame tall,
    # is 0.1 body heights whatever the camera distance.
    ts, ankles, heights, wrists = _clip(stance_ankle=0.94, event_ankle=0.908)
    assert abs(takeoff_elevation(ts, ankles, heights, wrists, 0) - 0.1) < 1e-6


def test_a_further_away_player_yields_the_same_ratio():
    """Perspective must cancel: the same jump, filmed from twice as far."""
    near = takeoff_elevation(*_clip(0.94, 0.908, body_height=0.32), 0)
    far = takeoff_elevation(*_clip(0.94, 0.924, body_height=0.16), 0)
    assert abs(near - far) < 1e-6


def test_no_stance_means_not_measured_rather_than_zero():
    """A recording opening mid-shot has no floor to measure against.

    Zero would be a claim that the player stayed down. None is the truth.
    """
    ts, ankles, heights, wrists = _clip(stance_ankle=0.94, event_ankle=0.90)
    # Keep only the event itself, as if the clip started there.
    keep = [i for i, t in enumerate(ts) if t >= -400]
    assert (
        takeoff_elevation(
            [ts[i] for i in keep],
            [ankles[i] for i in keep],
            [heights[i] for i in keep],
            [wrists[i] for i in keep],
            0,
        )
        is None
    )


def test_unseen_ankles_do_not_count_towards_the_stance():
    """Frames where the ankles failed the visibility gate cannot define a floor.

    Blinding every frame before the shot leaves the hand-low frames intact, so
    the stance is still located -- but with no ankle in any of them there is
    nothing to take a median of, and the answer must be None rather than a
    floor inferred from the shot itself.
    """
    ts, ankles, heights, wrists = _clip(0.94, 0.90)
    blinded = [None if t < -400 else a for t, a in zip(ts, ankles)]
    assert takeoff_elevation(ts, blinded, heights, wrists, 0) is None


def test_a_body_too_small_to_measure_is_skipped():
    ts, ankles, _, wrists = _clip(0.94, 0.90)
    tiny = [0.02] * len(ts)
    assert takeoff_elevation(ts, ankles, tiny, wrists, 0) is None


def test_shooting_event_is_the_wrist_peak():
    ts, _, _, wrists = _clip(0.94, 0.90)
    event = shooting_event_ms(ts, wrists)
    assert event is not None and abs(event) <= 40


def test_shooting_event_ignores_unseen_wrists():
    assert shooting_event_ms([0, 33, 66], [None, None, None]) is None


# --- measured values, pinned -------------------------------------------------
#
# Every set shot below sits under the threshold and the jump shot above it,
# with the 0.12 threshold unchanged. Sources: video8 shots 02-10 against the
# frame-exact manifest, and salah_video shots 1-5.

MEASURED_SET = [0.015, 0.018, 0.023, 0.036, 0.025, 0.050, 0.054, 0.100,
                0.087, 0.101, 0.075, 0.010, 0.070]
MEASURED_JUMP = [0.181]


def test_every_measured_set_shot_classifies_as_a_set_shot():
    for value in MEASURED_SET:
        result = classify(
            AttemptEvidence(
                vertical_displacement_m=value,
                horizontal_travel_m=0.05,
                reached_release=True,
                duration_s=1.5,
            )
        )
        assert result.shot_type is ShotType.SET_SHOT, value


def test_every_measured_jump_shot_classifies_as_a_jump_shot():
    for value in MEASURED_JUMP:
        result = classify(
            AttemptEvidence(
                vertical_displacement_m=value,
                horizontal_travel_m=0.05,
                reached_release=True,
                duration_s=1.5,
            )
        )
        assert result.shot_type is ShotType.JUMP_SHOT, value


def test_the_threshold_still_separates_the_two_measured_groups():
    """Guards the margin, not just the verdict.

    A change that leaves every label correct but erases the headroom above the
    highest observed set shot has made the classifier brittle without failing
    any other test here.
    """
    assert max(MEASURED_SET) < JUMP_VERTICAL_DISPLACEMENT_RATIO < min(MEASURED_JUMP)
    assert JUMP_VERTICAL_DISPLACEMENT_RATIO - max(MEASURED_SET) >= 0.015


def test_unmeasured_elevation_still_answers_but_with_capped_confidence():
    """A clip cut at the shot has no stance, and still deserves an answer.

    Refusing would discard every single-shot clip, which the whole-clip
    reference classifies correctly. The answer is marked provisional instead.
    """
    provisional = classify(
        AttemptEvidence(
            vertical_displacement_m=0.02,
            vertical_displacement_measured=False,
            horizontal_travel_m=0.05,
            reached_release=True,
            duration_s=1.5,
        )
    )
    measured = classify(
        AttemptEvidence(
            vertical_displacement_m=0.02,
            horizontal_travel_m=0.05,
            reached_release=True,
            duration_s=1.5,
        )
    )
    assert provisional.shot_type is ShotType.SET_SHOT
    assert provisional.confidence <= UNMEASURED_CONFIDENCE_CEILING
    assert provisional.confidence < measured.confidence
    assert any("provisional" in note for note in provisional.evidence)


def test_minimum_stance_samples_is_enforced():
    assert MIN_BASELINE_SAMPLES >= 5
