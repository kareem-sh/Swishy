"""
Regression tests pinning Swichy's canonical world-space convention: +Y UP.

BACKGROUND
----------
MediaPipe's `pose_world_landmarks` are emitted with +Y pointing DOWN. Every
consumer in this codebase used to assume +Y UP, and the old hand-typed unit
test fixtures *also* encoded +Y UP by hand -- so the test suite was green
while the product was measurably broken (the trunk-lean rule passed 0 of
1359 real recorded frames). Green tests were validating a coordinate
convention that never existed on the wire.

THE FIX
-------
`pose/landmarks.py` defines::

    MEDIAPIPE_TO_SWICHY = np.array([1.0, -1.0, 1.0])

and `extract_world_landmarks` multiplies every raw MediaPipe world position
by this vector, so Swichy's canonical world space is +Y UP from that point
on. Image-space landmarks are deliberately left un-flipped (screen
convention: +Y grows downward).

These tests are driven by REAL recorded model output, not hand-typed
fixtures, so they cannot repeat the original mistake of encoding an
assumption instead of measuring one.

FIXTURE PROVENANCE
------------------
`tests/fixtures/real_world_landmarks.json` is recorded from the live
MediaPipe model via `scripts/build_test_fixtures.py`. It must be
regenerated with that script -- NEVER hand-edited. Hand-editing it would
silently reintroduce the exact failure mode these tests exist to prevent
(a fixture that encodes what a human *assumes* rather than what the model
*emits*).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from angles.calculator import AngleCalculator
from angles.joint_chains import JOINT_CHAINS
from geometry.vectors import angle_between_vectors, angle_from_vertical, midpoint, segment_vector
from phase_detection.features import extract_features
from pose.landmarks import MEDIAPIPE_TO_SWICHY, extract_all_landmarks
from pose.visibility import VisibilityGate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "real_world_landmarks.json"

# The fixture does not encode handedness, so every AngleCalculator /
# extract_features call in this module fixes "right" consistently.
SHOOTING_SIDE = "right"

# Physically plausible trunk lean for someone standing or shooting. The
# pre-fix code returned 165-180 deg for exactly these frames.
MAX_PLAUSIBLE_TRUNK_LEAN_DEG = 45.0


def _load_fixture_raw():
    if not FIXTURE_PATH.exists():
        return None
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


_FIXTURE_RAW = _load_fixture_raw()
_SAMPLES = _FIXTURE_RAW["samples"] if _FIXTURE_RAW else []

# When the fixture is missing, parametrize with a single None sentinel so the
# failure surfaces as one clearly-labelled skip instead of silently collecting
# zero tests.
_PARAM_SAMPLES = _SAMPLES if _SAMPLES else [None]
_PARAM_IDS = (
    [f"{s['video']}::frame{s['frame_index']}({s.get('label', '')})" for s in _SAMPLES]
    if _SAMPLES
    else ["fixture-missing"]
)

_MISSING_MSG = (
    f"missing fixture file: {FIXTURE_PATH}. Regenerate it with "
    "scripts/build_test_fixtures.py -- do not hand-write it."
)


@pytest.fixture(scope="module")
def fixture_data():
    """The real, model-recorded landmark fixture, loaded once for the module."""
    if _FIXTURE_RAW is None:
        pytest.skip(_MISSING_MSG)
    return _FIXTURE_RAW


def _skip_if_sentinel(sample):
    if sample is None:
        pytest.skip(_MISSING_MSG)


def _avg_y(world, names):
    vals = [world[n]["position"][1] for n in names if n in world]
    assert vals, f"expected at least one of {names} present"
    return sum(vals) / len(vals)


def _make_gate():
    return VisibilityGate(visibility_threshold=0.6, presence_threshold=0.5)


def _sample_to_gated_world(sample, gate, negate=False):
    """Shape a fixture sample into the dict form AngleCalculator expects."""
    sign = -1.0 if negate else 1.0
    raw = {
        name: {
            "position": np.array(entry["position"], dtype=float) * sign,
            "visibility": entry.get("visibility", 1.0),
            "presence": entry.get("presence", 1.0),
        }
        for name, entry in sample["world"].items()
    }
    return gate.apply(raw)


def _synthetic_landmarks(wrist_y, index_y):
    """A minimal upright right-handed pose in canonical (+Y up) space."""
    positions = {
        "nose": [0.00, 1.65, 0.00],
        "left_shoulder": [-0.20, 1.45, 0.00],
        "right_shoulder": [0.20, 1.45, 0.00],
        "left_elbow": [-0.25, 1.15, 0.00],
        "right_elbow": [0.35, 1.20, 0.05],
        "left_wrist": [-0.25, 0.95, 0.00],
        "right_wrist": [0.40, wrist_y, 0.10],
        "left_index": [-0.25, 0.90, 0.00],
        "right_index": [0.42, index_y, 0.12],
        "left_hip": [-0.15, 0.95, 0.00],
        "right_hip": [0.15, 0.95, 0.00],
        "left_knee": [-0.15, 0.55, 0.00],
        "right_knee": [0.15, 0.55, 0.00],
        "left_ankle": [-0.15, 0.10, 0.00],
        "right_ankle": [0.15, 0.10, 0.00],
        "left_heel": [-0.15, 0.05, -0.05],
        "right_heel": [0.15, 0.05, -0.05],
        "left_foot_index": [-0.15, 0.05, 0.15],
        "right_foot_index": [0.15, 0.05, 0.15],
    }
    raw = {
        name: {"position": np.array(pos, dtype=float), "visibility": 1.0, "presence": 1.0}
        for name, pos in positions.items()
    }
    gate = _make_gate()
    return gate.apply(raw), gate


# ---------------------------------------------------------------------------
# 0. Fixture sanity
# ---------------------------------------------------------------------------

def test_fixture_has_samples(fixture_data):
    """Guards the guard.

    If the fixture were empty or malformed, every parametrized test below
    would skip one-by-one instead of failing loudly. Also pins that the
    fixture is labelled with the convention it claims to hold.
    """
    assert len(fixture_data["samples"]) > 0
    assert fixture_data.get("convention") == "swichy_canonical_y_up"


# ---------------------------------------------------------------------------
# 1. The ordering invariant -- the test that would have caught the bug
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample", _PARAM_SAMPLES, ids=_PARAM_IDS)
def test_anatomical_vertical_ordering(sample):
    """Pins that real recorded landmarks put the head above the feet.

    In canonical (+Y up) space every sample must satisfy
    nose > shoulders > hips > knees > ankles along Y. Under the raw
    MediaPipe (+Y down) convention this ordering is exactly inverted, so
    this single assertion would have caught the original bug immediately
    instead of the trunk-lean rule silently failing on 1359 real frames.
    """
    _skip_if_sentinel(sample)
    world = sample["world"]

    nose_y = world["nose"]["position"][1]
    shoulder_y = _avg_y(world, ["left_shoulder", "right_shoulder"])
    hip_y = _avg_y(world, ["left_hip", "right_hip"])
    knee_y = _avg_y(world, ["left_knee", "right_knee"])
    ankle_y = _avg_y(world, ["left_ankle", "right_ankle"])

    where = f"{sample['video']} frame {sample['frame_index']} ({sample.get('label')})"
    assert nose_y > shoulder_y, f"nose not above shoulders in {where}"
    assert shoulder_y > hip_y, f"shoulders not above hips in {where}"
    assert hip_y > knee_y, f"hips not above knees in {where}"
    assert knee_y > ankle_y, f"knees not above ankles in {where}"


# ---------------------------------------------------------------------------
# 2. The axis constant
# ---------------------------------------------------------------------------

def test_mediapipe_to_swichy_is_exact_y_negation():
    """Pins that the fix is exactly a Y sign flip, and is its own inverse.

    Because the transform is a reflection, applying it twice round-trips to
    raw MediaPipe coordinates rather than compounding -- so a stray double
    flip shows up as "the bug came back", not as a new failure mode.
    """
    assert MEDIAPIPE_TO_SWICHY.shape == (3,)
    np.testing.assert_array_equal(MEDIAPIPE_TO_SWICHY, np.array([1.0, -1.0, 1.0]))

    raw = np.array([0.31, -0.87, 1.42])
    once = raw * MEDIAPIPE_TO_SWICHY
    twice = once * MEDIAPIPE_TO_SWICHY

    assert once[0] == raw[0]
    assert once[1] == -raw[1]
    assert once[2] == raw[2]
    np.testing.assert_allclose(twice, raw)


# ---------------------------------------------------------------------------
# 3. angle_from_vertical semantics
# ---------------------------------------------------------------------------

def test_angle_from_vertical_basic_semantics():
    """Pins that [0,1,0] is the zero-degree reference.

    Straight up reads ~0 deg, horizontal ~90 deg, straight down ~180 deg.
    If this regresses to assuming +Y-down input, up and down silently swap.
    """
    assert angle_from_vertical(np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert angle_from_vertical(np.array([1.0, 0.0, 0.0])) == pytest.approx(90.0, abs=1e-6)
    assert angle_from_vertical(np.array([0.0, -1.0, 0.0])) == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("sample", _PARAM_SAMPLES, ids=_PARAM_IDS)
def test_real_trunk_segment_lean_is_plausible(sample):
    """Pins the exact defect this module exists to prevent.

    A real hip-to-shoulder segment must read as a small, physically
    plausible lean. Before the fix the same real segment returned roughly
    165-180 deg -- angle_from_vertical read an upright player as nearly
    upside down, which is why the trunk rule passed 0 of 1359 frames.
    """
    _skip_if_sentinel(sample)
    world = sample["world"]

    hip_mid = midpoint(
        np.array(world["left_hip"]["position"], dtype=float),
        np.array(world["right_hip"]["position"], dtype=float),
    )
    shoulder_mid = midpoint(
        np.array(world["left_shoulder"]["position"], dtype=float),
        np.array(world["right_shoulder"]["position"], dtype=float),
    )
    lean = angle_from_vertical(segment_vector(hip_mid, shoulder_mid))

    where = f"{sample['video']} frame {sample['frame_index']}"
    assert 0.0 <= lean <= MAX_PLAUSIBLE_TRUNK_LEAN_DEG, (
        f"implausible trunk lean {lean:.1f} deg for {where}; "
        "pre-fix (+Y-down) code returned ~165-180 deg here"
    )


# ---------------------------------------------------------------------------
# 4. Trunk angle through the production path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample", _PARAM_SAMPLES, ids=_PARAM_IDS)
def test_trunk_angle_via_angle_calculator_is_plausible(sample):
    """Pins the end-to-end production path, not just the geometry helper.

    Real landmarks through VisibilityGate then AngleCalculator.compute_all
    must yield a valid trunk result with a plausible lean. This is the
    fixed version of the rule that used to pass 0 of 1359 real frames.
    """
    _skip_if_sentinel(sample)
    gate = _make_gate()
    world = _sample_to_gated_world(sample, gate)
    results = AngleCalculator(gate).compute_all(world, SHOOTING_SIDE)

    assert "trunk" in results
    trunk = results["trunk"]
    where = f"{sample['video']} frame {sample['frame_index']}"
    assert trunk.is_valid, f"trunk angle invalid for {where}"
    assert 0.0 <= trunk.degrees <= MAX_PLAUSIBLE_TRUNK_LEAN_DEG, (
        f"implausible trunk angle {trunk.degrees:.1f} deg for {where}"
    )


# ---------------------------------------------------------------------------
# 5. Reflection invariance of three-landmark joint angles
# ---------------------------------------------------------------------------

def test_angle_between_vectors_reflection_invariance():
    """Pins the linear algebra that contained the blast radius.

    cos(theta) = (u.v)/(|u||v|). Negating both u and v leaves the dot
    product and both magnitudes unchanged, so the angle is identical. This
    is why elbow/knee/hip/shoulder angles were never affected by the bug.
    """
    u = np.array([0.3, 0.7, -0.2])
    v = np.array([-0.5, 0.1, 0.9])
    assert angle_between_vectors(-u, -v) == pytest.approx(angle_between_vectors(u, v), abs=1e-6)


@pytest.mark.parametrize("sample", _PARAM_SAMPLES, ids=_PARAM_IDS)
def test_joint_angles_are_reflection_invariant_on_real_data(sample):
    """Pins the asymmetry that let the bug hide, using real data.

    Three-landmark joint angles are unchanged when the whole frame is
    negated; trunk lean is not, because it is measured against a fixed
    external axis. That asymmetry is precisely why the old +Y-up-assuming
    suite stayed green while trunk lean was silently broken.
    """
    _skip_if_sentinel(sample)
    gate = _make_gate()
    calc = AngleCalculator(gate)

    upright = calc.compute_all(_sample_to_gated_world(sample, gate), SHOOTING_SIDE)
    reflected = calc.compute_all(
        _sample_to_gated_world(sample, _make_gate(), negate=True), SHOOTING_SIDE
    )

    where = f"{sample['video']} frame {sample['frame_index']}"
    compared = 0
    for key in JOINT_CHAINS:
        a, b = upright.get(key), reflected.get(key)
        if a is None or b is None or not (a.is_valid and b.is_valid):
            continue
        compared += 1
        assert a.degrees == pytest.approx(b.degrees, abs=1e-6), (
            f"{key} was not reflection-invariant for {where}"
        )

    assert compared > 0, f"no comparable joint angles for {where}; test would be vacuous"


# ---------------------------------------------------------------------------
# 6. Velocity sign semantics
# ---------------------------------------------------------------------------

def _wrist_velocity(prev_wrist_y, curr_wrist_y):
    world_prev, gate = _synthetic_landmarks(wrist_y=prev_wrist_y, index_y=prev_wrist_y - 0.05)
    calc = AngleCalculator(gate)
    angles_prev = calc.compute_all(world_prev, SHOOTING_SIDE)

    world_curr, _ = _synthetic_landmarks(wrist_y=curr_wrist_y, index_y=curr_wrist_y - 0.05)
    angles_curr = calc.compute_all(world_curr, SHOOTING_SIDE)

    return extract_features(
        world_curr,
        angles_curr,
        SHOOTING_SIDE,
        prev_world=world_prev,
        prev_angles=angles_prev,
        dt_s=1.0 / 30.0,
    ).wrist_velocity_y


def test_wrist_velocity_positive_when_wrist_moves_up():
    """Pins that rising equals positive vertical velocity.

    Getting this sign backwards makes ball-lift and release detection fire
    on the descending half of the shooting motion, which is what happened
    before the fix.
    """
    assert _wrist_velocity(0.90, 1.00) > 0.0


def test_wrist_velocity_negative_when_wrist_moves_down():
    """Pins the opposite direction: falling equals negative velocity."""
    assert _wrist_velocity(0.90, 0.80) < 0.0


# ---------------------------------------------------------------------------
# 7. Image landmarks are deliberately NOT flipped
# ---------------------------------------------------------------------------

def test_image_landmarks_are_not_flipped():
    """Pins that only WORLD landmarks are flipped, never image landmarks.

    Image landmarks feed the drawing code, which expects screen coordinates
    where +Y grows downward. So the nose must have a SMALLER image y than
    the ankle. If the flip ever leaks into extract_image_landmarks, the
    skeleton overlay renders upside down.
    """
    image_path = PROJECT_ROOT / "assets" / "test.jpg"
    if not image_path.exists():
        pytest.skip(f"sample image not available: {image_path}")

    cv2 = pytest.importorskip("cv2")
    mp = pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import vision

    from pose.detector import PoseDetector

    image = cv2.imread(str(image_path))
    if image is None:
        pytest.skip(f"could not decode {image_path}")

    detector = PoseDetector(vision.RunningMode.IMAGE)
    result = detector.detect_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = extract_all_landmarks(result, image.shape[1], image.shape[0])
    if landmarks is None:
        pytest.skip("no pose detected in sample image")

    nose_screen_y = landmarks["image"]["nose"]["y"]
    ankle_screen_y = landmarks["image"]["right_ankle"]["y"]
    assert nose_screen_y < ankle_screen_y, (
        "image-space nose y should be smaller (higher on screen) than ankle y; "
        "image landmarks must NOT be flipped by MEDIAPIPE_TO_SWICHY"
    )

    # And the world landmarks from the same detection must be the other way up.
    assert landmarks["world"]["nose"]["position"][1] > landmarks["world"]["right_ankle"]["position"][1]
