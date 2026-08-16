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
from shots.types import RejectionReason, ShotType  # noqa: E402

VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"

VIDEO_01 = VIDEO_DIR / "video_01_free_throw.mp4"
VIDEO_07 = VIDEO_DIR / "video_07_side_jump_shot.mp4"

# Ground truth from the footage in both cases.
#
# video_01 was re-checked frame by frame rather than inherited from the
# pipeline: the ball goes chest -> overhead -> away twice, at roughly 2.5 s and
# 11 s. Two attempts. What happens in between -- the ball returning at 8.5 s and
# being caught at 9.5 s -- is not a third. See _CATCH_LIMITATION.
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
# --------------------------------------------------------------------------
# Whole-video multi-shot fixtures.
#
# The single-shot clips below prove the pipeline scores ONE attempt correctly.
# They cannot prove it separates several attempts in one recording, because
# each clip is cut to contain exactly one. These two fixtures close that gap.
#
# Ground truth is the same footage the clips were cut from:
#   video8.mov  10 attempts (9 set shots, then one low jump)
#   video9.mov   3 attempts (all jump shots)
VIDEO_08_FULL = VIDEO_DIR / "video8.mov"
VIDEO_09_FULL = VIDEO_DIR / "video9.mov"
EXPECTED_SHOTS_VIDEO_08 = 10
EXPECTED_SHOTS_VIDEO_09 = 3

# MOSTLY RESOLVED — under-segmentation of continuous shooting.
#
# This was the big one: video8 yielded 4 of 10 and video9 1 of 3, while every
# clip cut from those same videos passed individually. The failure was in
# separating consecutive attempts, not in analysing one.
#
# The cause was a sequence that could not fit rather than a threshold that
# needed tuning. An attempt could only start from rest, so recovery after each
# shot (post-release window, then the walk back to rest, then a refractory
# guard: 3-5 s in total) had to finish before the next attempt could open. The
# gap between attempts in continuous practice footage is about 2.4 s. Every
# recovery swallowed the next shot, and four separate retunings each recovered
# one fixture by breaking another.
#
# Fixed by anchoring attempts on the release and rebuilding the shot window
# backwards out of the frame buffer, so whether the tracker was "ready" stopped
# mattering. Measured after: video9 3 of 3, video8 9 of 10.
#
# strict=True: if a later change makes the remaining one pass, the suite fails
# and says so, instead of letting the limitation be quietly forgotten.
_MULTI_SHOT_LIMITATION = (
    "video8 yields 9 of 10 attempts; one attempt is still not separated from "
    "its neighbour"
)

# KNOWN FALSE POSITIVE — catching is not separable from shooting by pose alone.
#
# video_01 contains two free throws, confirmed from the footage (the ball goes
# from chest to overhead and away at ~2.5 s and ~11 s). Four attempts are
# reported. The two extras are:
#
#   0.0-1.5 s   the recording opens with the arms already overhead -- the
#               follow-through of a shot that began before the camera did.
#   6.6-9.1 s   the player raises both hands to CATCH the returning ball.
#               0.670 m of wrist travel, hands below shoulder to above and back
#               down: kinematically identical to a shot.
#
# Requiring the elbow to extend was tried, on the reasoning that a shot
# straightens the arm while a catch collapses it to absorb. Reaching up for a
# ball extends the arm too, so it did not separate them -- and it cost a real
# attempt in video8 (9 detected -> 8). Reverted.
#
# The signal that actually separates them is the BALL: moving away from the
# hands, or towards them. That is exactly what the ball tracker adds, so this
# is a gap with a known owner rather than an open question.
_CATCH_LIMITATION = (
    "a catch is hands-up-then-down with an extended arm, the same pose as a "
    "shot; separating them needs the ball, not the body"
)

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
    ("video9_shot03_jump.mp4", ShotType.JUMP_SHOT),
]

# video9_shot01 and video9_shot02 are NOT in the list above, and this is a
# statement about the clips rather than about the code.
#
# Both were cut after the shot had already begun. Measured, as the shooting
# hand's height at frame zero against its own peak in that clip:
#
#     video9_shot01   opens at 81% of its peak    prominence 0.113   not found
#     video9_shot02   opens at 94% of its peak    prominence 0.091   not found
#     video9_shot03   opens at 33% of its peak    prominence 0.671   found
#     video8_shot05   opens at 33% of its peak    prominence 0.572   found
#
# A shot is located by how far the hand rises above where it rested
# beforehand. In these two clips those frames were never recorded, so there is
# no rise to measure and no threshold that could recover one -- lowering the
# bar far enough to catch a prominence of 0.09 would call ordinary movement a
# shot in every other video.
#
# They are removed rather than xfailed because an xfail asserts the code is
# wrong and should one day pass. Here the code is right: the correct behaviour
# on a clip cut mid-shot is to decline it and say why, which
# shots.segmenter.explain_absence now does. That behaviour is tested below.
_CLIPS_CUT_MID_SHOT = ["video9_shot01_jump.mp4", "video9_shot02_jump.mp4"]

_cache = {}


def _run(path: Path):
    """Analyse a fixture once per session; skip cleanly if it is not present.

    `enable_ball=False` is stated here on purpose. These tests are about pose,
    segmentation and scoring, and the ball detector changes when a shot is
    considered finished (the tracker waits for a ball outcome instead of
    closing on body motion), which would make them measure two things at once.
    It also loads YOLO, which is slow. Until this was explicit the same value
    was inherited by accident from a hardcoded default in the CLI.
    """
    if not path.exists():
        pytest.skip(f"fixture not available: {path.name}")
    if path not in _cache:
        _cache[path] = analyze_video(path, enable_ball=False)
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
# RESOLVED, and what replaced it.
#
# This clip used to be rejected outright: the credibility gate never saw the
# shooting wrist above the shoulder, so a real attempt could not be confirmed.
# That is fixed. Release-anchored segmentation rebuilds the shot window
# backwards from the release instead of forwards from a posture change, and the
# attempt is now found, scored, and analysed.
#
# THE OLD EXPLANATION WAS ALSO WRONG, and it is worth recording why, because it
# sent the search in the wrong direction for a while. The failure was blamed on
# "480x360 with a distant player, roughly a sixth the pixel area of the other
# fixtures". Measured, the player fills MORE of this frame than of any other
# multi-shot fixture -- 0.463 of frame height against 0.212 for video8 and
# 0.231 for video9. Low absolute resolution was never the problem.
#
# WHAT REMAINS is one specific measurement, and it is narrower than it looks.
# `body_rise_ratio` reads a peak of 0.031 body heights on this clip against
# 0.496 and 0.500 on video8 and video9 -- sixteen times smaller on a shot that
# is visibly a jump. Everything else about the clip measures normally. The
# leading hypothesis is the standing ankle baseline: it adapts per FRAME, and
# this is the only slow-motion fixture, so the baseline has many more frames in
# which to creep upward during the flight and cancel the very rise it exists to
# measure. That would make it the last per-frame constant in a codebase that is
# otherwise per-second throughout -- the same class of bug already fixed in the
# detector, the tracker and the segmentation config.
#
# Not fixed here because it is a change to a signal every fixture depends on,
# and it deserves its own measurement pass rather than being folded into a
# segmentation change.
#
# strict=True so that if a future change makes this pass, the suite fails and
# says so rather than letting the limitation be quietly forgotten.
_SLOW_MOTION_RISE_LIMITATION = (
    "slow motion: the ankle baseline adapts per frame, so it creeps up during "
    "the longer flight and body_rise_ratio reads 0.031 against 0.50 on "
    "comparable fixtures -- the jump is real but unmeasured, so the classifier "
    "sees a stationary shot."
)


@pytest.fixture(scope="module")
def video_07_run():
    return _run(VIDEO_07)


@pytest.mark.xfail(strict=True, reason=_SLOW_MOTION_RISE_LIMITATION)
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


def test_video_07_is_analysed_not_rejected(video_07_run):
    assert not video_07_run.is_rejected, _describe(video_07_run)


def test_video_07_detects_expected_shot_count(video_07_run):
    assert len(video_07_run.shots) == EXPECTED_SHOTS_VIDEO_07, _describe(video_07_run)


@pytest.mark.xfail(strict=True, reason=_SLOW_MOTION_RISE_LIMITATION)
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


@pytest.mark.xfail(strict=True, reason=_CATCH_LIMITATION)
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


# ------------------------------------------------- whole-video multi-shot


@pytest.fixture(scope="module")
def video_08_full_run():
    return _run(VIDEO_08_FULL)


@pytest.fixture(scope="module")
def video_09_full_run():
    return _run(VIDEO_09_FULL)


def test_video_08_full_never_reports_a_driving_action(video_08_full_run):
    """A fixed-camera practice video contains no driving actions.

    Every attempt here is a stationary shot, so every attempt must classify as
    one of the two supported types. Anything refused as an unsupported type
    means the driving gate fired on a player who never drove.

    HISTORY: this was xfail(strict) because one candidate spanned two attempts
    plus the walk between them, and that walk pushed hip travel over the 0.18
    driving threshold. Restricting the travel window to the rise and anchor
    phases (feedback/shot_tracker.py:_TRAVEL_STATES) fixed it.

    The merging itself still happens -- test_video_08_full_detects_every_attempt
    is xfail for exactly that -- so this test passes because the merged
    candidate no longer CROSSES the threshold, not because candidates stopped
    merging. Widening _TRAVEL_STATES, or lowering the threshold, would break it
    again.

    Asserted on the rejection rather than on the string "layup": the driving
    branch reports UNKNOWN, so a label check would pass no matter what the gate
    did, and a test that cannot fail is worse than no test.
    """
    offenders = [
        s for s in video_08_full_run.shots
        if s.rejection is RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET
    ]
    assert not offenders, (
        "the player never drives in this fixture; a candidate that spans the "
        "walk between attempts was misread as a moving action.\n"
        f"{_describe(video_08_full_run)}"
    )


@pytest.mark.xfail(strict=True, reason=_MULTI_SHOT_LIMITATION)
def test_video_08_full_detects_every_attempt(video_08_full_run):
    assert len(video_08_full_run.shots) == EXPECTED_SHOTS_VIDEO_08, _describe(
        video_08_full_run
    )


def test_video_09_full_detects_every_attempt(video_09_full_run):
    assert len(video_09_full_run.shots) == EXPECTED_SHOTS_VIDEO_09, _describe(
        video_09_full_run
    )


def test_video_09_full_shots_are_all_jump_shots(video_09_full_run):
    """Whatever subset is found, none of it may be called a set shot."""
    for shot in video_09_full_run.shots:
        assert shot.shot_type is ShotType.JUMP_SHOT, (
            f"expected every detected shot to be {ShotType.JUMP_SHOT.value}\n"
            f"{_describe(video_09_full_run)}"
        )


def test_multi_shot_videos_do_not_leak_state_between_shots(video_08_full_run):
    """Shot numbers are sequential and each shot carries its own timing."""
    shots = video_08_full_run.shots
    assert [s.shot_number for s in shots] == list(range(1, len(shots) + 1))
    scored = [s for s in shots if not s.is_rejected]
    starts = [s.start_timestamp_ms for s in scored if s.start_timestamp_ms is not None]
    assert starts == sorted(starts), "attempts are not in chronological order"
    for shot in scored:
        if shot.start_timestamp_ms is None or shot.end_timestamp_ms is None:
            continue
        assert shot.end_timestamp_ms > shot.start_timestamp_ms


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


@pytest.mark.parametrize("clip", _CLIPS_CUT_MID_SHOT)
def test_clip_cut_mid_shot_is_declined_with_a_usable_reason(clip):
    """Declining is the right answer here. Declining SILENTLY is not.

    These clips open with the hand already most of the way up, so there is no
    rise to measure. The system cannot analyse them and should not pretend to.

    What it must do is say which of several very different things went wrong,
    because each one means something different for the person filming: start
    recording earlier, improve the lighting, keep the player in frame. "No shot
    was found" is true and tells them none of that.
    """
    run = _run(SINGLE_SHOT_DIR / clip)
    assert not run.shots, "expected no shot on a clip cut mid-shot"

    reason = run.no_shot_reason or run.rejection_detail
    assert reason, "declined without saying why"
    assert "already" in reason.lower() and "record" in reason.lower(), (
        "the reason should identify the clip as starting mid-shot and say what "
        f"to do about it, but said:\n  {reason}"
    )
