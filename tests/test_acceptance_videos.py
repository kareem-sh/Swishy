"""
End-to-end acceptance tests: real video in, real classification out.

WHY THIS FILE EXISTS
--------------------
The rest of the suite tests components in isolation, and that hid a real bug.
`test_feedback.py::test_set_shot_and_jump_shot_are_distinguished` hands the
tracker a synthetic ``ankle_rise=0.20`` and asserts the classifier answers
JUMP_SHOT. That proves the classifier's arithmetic and nothing more:

    ankle_rise=0.20  ->  classifier  ->  JUMP_SHOT        (already tested)

    video_07 -> pose -> features -> tracker -> classifier -> JUMP_SHOT
                                                            (NOT tested)

Every link in the second chain was unverified, so a suite that was entirely
green could not see that video_07 -- a real jump shot -- is classified as a
set shot. These tests close that gap by running the acceptance fixtures
through the production pipeline with nothing mocked.

WHAT IS ASSERTED
----------------
Only the externally meaningful contract: a shot was found, it was called the
right thing, and it produced a score with phase analysis. Exact angles and
scores are deliberately NOT asserted -- those are covered by unit tests, and
pinning them here would make this file break on every legitimate tuning
change instead of on real regressions.

These tests decode entire videos through MediaPipe, so they are slow by
nature. Each fixture is analysed once per session and cached.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from coach_report import analyze_video  # noqa: E402
from shots.types import ShotType  # noqa: E402

VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"

VIDEO_01 = VIDEO_DIR / "video_01_free_throw.mp4"
VIDEO_07 = VIDEO_DIR / "video_07_side_jump_shot.mp4"

# Expected shot counts are the CURRENT measured behaviour of the fixtures,
# recorded so that a change in segmentation is visible rather than silent.
# video_07 went 7 -> 1 when candidate-based segmentation replaced the old
# "any posture change is a shot" rule; that 1 is the number being locked in.
EXPECTED_SHOTS_VIDEO_01 = 2
EXPECTED_SHOTS_VIDEO_07 = 1

# --------------------------------------------------------------------------
# Single-shot clips.
#
# video8 and video9 were cut into one clip per shooting attempt, so a failure
# names one shot instead of one video. Boundaries sit at the midpoint between
# adjacent attempts, so a clip cannot contain two shooting events.
#
# Ground truth comes from the FOOTAGE, not the pipeline. Each attempt was
# located with raw MediaPipe (hands above head) and then confirmed frame by
# frame:
#
#   video8 - 10 attempts, fixed wide camera, no cuts. In shots 1-9 the feet
#            never leave the floor: the player rises onto the toes and that is
#            all. Shot 10 IS a jump, just a low one -- the ankles trace a
#            clean ballistic arc, flat at image-y 0.7133, peaking at 0.6749,
#            back to 0.7116 over ~0.57 s.
#
#   video9 - 3 attempts, all unambiguously airborne. A fourth hands-overhead
#            event at 2.8 s is the player bending to collect the ball, not a
#            shot, and is deliberately excluded.
#
# The clips are generated, not committed (assets/videos/**/*.mp4 is ignored);
# assets/videos/single_shot/manifest.json records how to rebuild them.
SINGLE_SHOT_DIR = VIDEO_DIR / "single_shot"

SINGLE_SHOT_CLIPS = [
    ("video8_shot01_set.mp4", ShotType.SET_SHOT),
    ("video8_shot02_set.mp4", ShotType.SET_SHOT),
    ("video8_shot03_set.mp4", ShotType.SET_SHOT),
    ("video8_shot04_set.mp4", ShotType.SET_SHOT),
    ("video8_shot05_set.mp4", ShotType.SET_SHOT),
    ("video8_shot06_set.mp4", ShotType.SET_SHOT),
    ("video8_shot07_set.mp4", ShotType.SET_SHOT),
    ("video8_shot08_set.mp4", ShotType.SET_SHOT),
    ("video8_shot09_set.mp4", ShotType.SET_SHOT),
    ("video8_shot10_jump.mp4", ShotType.JUMP_SHOT),
    ("video9_shot01_jump.mp4", ShotType.JUMP_SHOT),
    ("video9_shot02_jump.mp4", ShotType.JUMP_SHOT),
    ("video9_shot03_jump.mp4", ShotType.JUMP_SHOT),
]

_cache = {}


def _run(path: Path):
    """Analyse a fixture once per session; skip cleanly if it is not present."""
    if not path.exists():
        pytest.skip(f"fixture not available: {path.name}")
    if path not in _cache:
        _cache[path] = analyze_video(path)
    return _cache[path]


def _describe(run) -> str:
    """Readable actual-state dump, so a failure explains itself."""
    if run.is_rejected:
        return f"video rejected: {run.rejection.value} — {run.rejection_detail}"
    parts = [
        f"{len(run.shots)} shot(s), "
        f"{run.discarded_candidates} discarded, "
        f"{run.frames_read} frames @ {run.fps:.2f} fps"
    ]
    for s in run.shots:
        kind = s.shot_type.value if s.shot_type else "none"
        parts.append(f"  shot {s.shot_number}: type={kind} score={s.score}")
    return "\n".join(parts)


# --------------------------------------------------------------- video_07
# The jump-shot fixture: one shooting motion, slow motion, side view.
#
# Ground truth was re-established from the footage -- raw MediaPipe found a
# single hands-overhead event at 18.77-24.21 s, confirmed visually -- so the
# expected count of 1 is not merely the pipeline's own past behaviour.
#
# KNOWN LIMITATION. These are xfail, not deleted and not weakened.
#
# video_07 is 480x360 with a distant player, roughly a sixth the pixel area of
# the other fixtures. The attempt IS found (5.47 s, 0.222 m of wrist travel),
# but the shooting wrist never rises above the shoulder in the landmarks: its
# peak is 0.194 body heights BELOW it, which is anatomically impossible for a
# real shot and is the clearest evidence that the pose estimate is unreliable
# at this resolution. The credibility gate therefore rejects the attempt.
#
# That is the right trade. The gate is what stops preparation and fidgeting
# from being scored; relaxing it far enough to admit this clip would admit any
# movement where the hands stay near the waist. Tuning it to this one file is
# precisely the fixture-specific fix that is not allowed.
#
# strict=True so that if a future change makes these pass, the suite fails and
# says so rather than letting the limitation be quietly forgotten.
_LOW_RES_LIMITATION = (
    "480x360 footage: the shooting wrist is never resolved above the shoulder, "
    "so a real attempt cannot pass the credibility gate."
)


@pytest.fixture(scope="module")
def video_07_run():
    return _run(VIDEO_07)


def test_video_07_is_never_scored_as_a_set_shot(video_07_run):
    """It may decline to score this clip, but it must not score it wrongly.

    The behaviour being guarded against is what this clip used to produce: a
    confident "Set Shot / Free Throw, 47/100" for a jump shot. Declining is
    honest; a confident wrong answer is not.
    """
    for shot in video_07_run.shots:
        assert shot.shot_type is not ShotType.SET_SHOT, (
            "video_07 is a jump shot; scoring it as a set shot is worse than "
            f"declining to score it.\n{_describe(video_07_run)}"
        )


@pytest.mark.xfail(strict=True, reason=_LOW_RES_LIMITATION)
def test_video_07_is_analysed_not_rejected(video_07_run):
    assert not video_07_run.is_rejected, _describe(video_07_run)


@pytest.mark.xfail(strict=True, reason=_LOW_RES_LIMITATION)
def test_video_07_detects_expected_shot_count(video_07_run):
    assert len(video_07_run.shots) == EXPECTED_SHOTS_VIDEO_07, _describe(video_07_run)


@pytest.mark.xfail(strict=True, reason=_LOW_RES_LIMITATION)
def test_video_07_is_classified_as_a_jump_shot(video_07_run):
    """video_07 shows a player leaving the floor. It is a jump shot.

    Exercises the whole chain: real frames -> pose landmarks -> image-space
    body-rise feature -> shot tracker evidence -> classifier.
    """
    shot = video_07_run.shots[0]
    assert shot.shot_type is ShotType.JUMP_SHOT, (
        f"expected: {ShotType.JUMP_SHOT.value}\n"
        f"actual:   {shot.shot_type.value if shot.shot_type else 'none'}\n"
        f"{_describe(video_07_run)}"
    )


@pytest.mark.xfail(strict=True, reason=_LOW_RES_LIMITATION)
def test_video_07_produces_a_score_and_phase_analysis(video_07_run):
    shot = video_07_run.shots[0]
    assert shot.score is not None, _describe(video_07_run)
    assert 0 <= shot.score <= 100
    assert shot.phase_scores, "no per-phase analysis was produced"


# --------------------------------------------------------------- video_01
# The set-shot / free-throw fixture. No jump; the Jump phase is legitimately
# absent and must not be penalised.


@pytest.fixture(scope="module")
def video_01_run():
    return _run(VIDEO_01)


def test_video_01_is_analysed_not_rejected(video_01_run):
    assert not video_01_run.is_rejected, _describe(video_01_run)


def test_video_01_detects_expected_shot_count(video_01_run):
    assert len(video_01_run.shots) == EXPECTED_SHOTS_VIDEO_01, _describe(video_01_run)


def test_video_01_is_classified_as_a_set_shot(video_01_run):
    for shot in video_01_run.shots:
        assert shot.shot_type is ShotType.SET_SHOT, (
            f"expected: {ShotType.SET_SHOT.value}\n"
            f"actual:   {shot.shot_type.value if shot.shot_type else 'none'}\n"
            f"{_describe(video_01_run)}"
        )


def test_video_01_produces_a_score_and_phase_analysis(video_01_run):
    for shot in video_01_run.shots:
        assert shot.score is not None, _describe(video_01_run)
        assert 0 <= shot.score <= 100
        assert shot.phase_scores, f"shot {shot.shot_number} has no phase analysis"


# ------------------------------------------------------ single-shot clips
# One clip, one shot. Three separable contracts, each its own test so a
# failure says WHICH stage broke rather than just "the clip failed":
#
#   1. exactly one shot is detected          (segmentation)
#   2. it is called the right thing          (classification)
#   3. it is scored with phase analysis      (evaluation)


@pytest.fixture(scope="module")
def clip_runs():
    """Analyse every clip once; each test then reads from the cache."""
    return {name: _run(SINGLE_SHOT_DIR / name) for name, _ in SINGLE_SHOT_CLIPS}


@pytest.mark.parametrize("clip,expected", SINGLE_SHOT_CLIPS)
def test_single_shot_clip_detects_exactly_one_shot(clip, expected, clip_runs):
    run = clip_runs[clip]
    assert len(run.shots) == 1, (
        f"expected: 1 shot\nactual:   {len(run.shots)}\n{_describe(run)}"
    )


@pytest.mark.parametrize("clip,expected", SINGLE_SHOT_CLIPS)
def test_single_shot_clip_is_classified_correctly(clip, expected, clip_runs):
    run = clip_runs[clip]
    if not run.shots:
        pytest.fail(f"no shot detected, so nothing to classify\n{_describe(run)}")
    actual = run.shots[0].shot_type
    assert actual is expected, (
        f"expected: {expected.value}\n"
        f"actual:   {actual.value if actual else 'none'}\n{_describe(run)}"
    )


@pytest.mark.parametrize("clip,expected", SINGLE_SHOT_CLIPS)
def test_single_shot_clip_is_scored(clip, expected, clip_runs):
    run = clip_runs[clip]
    if not run.shots:
        pytest.fail(f"no shot detected, so nothing to score\n{_describe(run)}")
    shot = run.shots[0]
    assert shot.score is not None, _describe(run)
    assert shot.phase_scores, "no per-phase analysis was produced"
