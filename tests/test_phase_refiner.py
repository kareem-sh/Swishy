"""Tests for post-hoc coaching-phase assignment.

The detector tracks four states so that shots are found reliably. These tests
cover the other half of that trade: that the seven coaching phases are still
recovered afterwards, from the captured frames, without the detector having
had to track any of them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedback.phase_refiner import compute_cuts, refine_phases
from phase_detection.features import KinematicFeatures
from phase_detection.phases import CORE_ANCHOR, PHASE_ORDER, phase_index
from utils.frame_buffer import FrameSnapshot


def _snap(t_ms, state, wrist_ratio, knee=None, rise=0.0, above_shoulder=None):
    return FrameSnapshot(
        timestamp_ms=t_ms,
        angles={},
        shooting_side="right",
        phase=state,
        features=KinematicFeatures(
            wrist_height_ratio=wrist_ratio,
            wrist_above_shoulder_ratio=(
                above_shoulder if above_shoulder is not None else wrist_ratio - 0.30
            ),
            knee_angle=knee,
            body_rise_ratio=rise,
        ),
    )


def _set_shot():
    """Stand, dip, lift, release, finish. The feet never leave the floor."""
    frames = []
    t = 0

    def add(state, wrist, knee, rise=0.0):
        nonlocal t
        frames.append(_snap(t, state, wrist, knee, rise))
        t += 33

    for knee in (175, 175, 174):          # standing
        add("ready", -0.05, knee)
    for knee in (160, 140, 118, 104):     # dipping
        add("rise", -0.08, knee)
    for wrist, knee in ((0.10, 120), (0.30, 145), (0.50, 165)):
        add("rise", wrist, knee)          # driving up
    for _ in range(2):
        add(CORE_ANCHOR, 0.62, 172)       # ball gone
    for wrist in (0.55, 0.40, 0.20, 0.05):
        add("recovery", wrist, 175)       # arm comes down
    return frames


def _jump_shot():
    """Same, but the whole body leaves the floor around the release."""
    frames = []
    t = 0

    def add(state, wrist, knee, rise):
        nonlocal t
        frames.append(_snap(t, state, wrist, knee, rise))
        t += 33

    for knee in (175, 174):
        add("ready", -0.05, knee, 0.0)
    for knee in (150, 120, 100):
        add("rise", -0.08, knee, 0.0)
    add("rise", 0.15, 130, 0.02)
    add("rise", 0.35, 155, 0.10)          # feet off the floor
    add("rise", 0.55, 170, 0.20)
    for _ in range(2):
        add(CORE_ANCHOR, 0.65, 174, 0.26)  # apex, ball gone
    add("recovery", 0.60, 174, 0.22)
    add("recovery", 0.45, 170, 0.08)      # coming down
    for wrist in (0.25, 0.15, 0.05, 0.00, -0.05, -0.05):
        add("recovery", wrist, 168, 0.0)  # back on the floor, arm dropping
    return frames


def test_a_set_shot_gets_the_ground_phases_and_no_jump():
    """No jump phase is the correct description of a set shot, not a gap."""
    labels = refine_phases(_set_shot())
    seen = set(labels)

    assert "loading" in seen
    assert "ball_lift" in seen
    assert "release" in seen
    assert "follow_through" in seen
    assert "jump" not in seen


def test_a_jump_shot_gets_the_airborne_phases():
    labels = refine_phases(_jump_shot())
    seen = set(labels)

    assert "jump" in seen, labels
    assert "landing" in seen, labels
    assert "release" in seen


def test_phases_run_forwards():
    """Whatever the signals do, the timeline may not go backwards.

    Real footage produces knee minima after the release (the legs tuck in
    flight) and wrist minima in odd places. A signal that reads out of sequence
    must collapse its own segment, never reorder the ones after it.
    """
    for frames in (_set_shot(), _jump_shot()):
        labels = refine_phases(frames)
        indices = [phase_index(p) for p in labels]
        assert indices == sorted(indices), labels


def test_the_deepest_knee_bend_lands_in_loading():
    """The loading rules score the dip, so the dip has to be inside loading.

    This is the whole reason boundaries are found by extrema rather than by
    live thresholds: locating the bottom of the dip afterwards is an argmin,
    which cannot get stuck or fire late.
    """
    frames = _set_shot()
    labels = refine_phases(frames)
    knees = [
        (f.features.knee_angle, i)
        for i, f in enumerate(frames)
        if f.features.knee_angle is not None
    ]
    _, deepest = min(knees)
    assert labels[deepest] == "loading", (labels, deepest)


def test_every_label_is_a_known_coaching_phase():
    for frames in (_set_shot(), _jump_shot()):
        for label in refine_phases(frames):
            assert label in PHASE_ORDER, label


def test_release_comes_from_the_state_machine_not_a_guess():
    """The anchor event defines the shot; the refiner must not second-guess it."""
    frames = _jump_shot()
    cuts = compute_cuts(frames)
    marked = [i for i, f in enumerate(frames) if f.phase == CORE_ANCHOR]
    assert cuts.release_start == marked[0]
    assert cuts.release_end == marked[-1] + 1


def test_empty_and_featureless_input_do_not_raise():
    assert refine_phases([]) == []
    bare = [
        FrameSnapshot(timestamp_ms=t, angles={}, shooting_side="right",
                      phase="ready", features=None)
        for t in (0, 33, 66)
    ]
    assert len(refine_phases(bare)) == 3


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
