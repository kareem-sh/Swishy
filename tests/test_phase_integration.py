"""Integration tests for the phase chain, end to end, nothing mocked.

WHAT THIS COVERS THAT THE UNIT TESTS DO NOT
-------------------------------------------
The phase model is now two layers that have to agree without ever being
compared directly:

    detector  ->  four live states       (phase_detection.detector)
    tracker   ->  candidate + back-fill  (feedback.shot_tracker)
    refiner   ->  seven coaching phases  (feedback.phase_refiner)
    engine    ->  rules keyed by phase   (analysis.engine, config/biomechanics)
    scorer    ->  per-phase scores       (feedback.scorer)

Each layer is unit-tested on its own, and each passed while the chain was
broken. The specific way it breaks silently is a NAME: a rule in
biomechanics.yaml keyed to a phase the refiner never emits simply never fires,
the shot still scores, and the only visible symptom is a coaching section
quietly missing from the report. Nothing raises.

These tests run real frames through the real pipeline and assert the chain
joins up.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from coach_report import analyze_video  # noqa: E402
from feedback.scorer import WHOLE_SHOT_PHASE  # noqa: E402
from phase_detection.phases import (  # noqa: E402
    CORE_STATES,
    PHASE_ORDER,
    phase_index,
)
from utils.config_loader import load_yaml  # noqa: E402

VIDEO_DIR = PROJECT_ROOT / "assets" / "videos"

# One of each: a single attempt, and continuous shooting with many.
SINGLE = VIDEO_DIR / "single_shot" / "video8_shot10_jump.mp4"
MULTI = VIDEO_DIR / "video9.mov"

_cache = {}


def _run(path: Path):
    if not path.exists():
        pytest.skip(f"fixture not available: {path.name}")
    if path not in _cache:
        _cache[path] = analyze_video(path, enable_ball=False)
    return _cache[path]


@pytest.fixture(scope="module")
def single_shot_run():
    return _run(SINGLE)


@pytest.fixture(scope="module")
def multi_shot_run():
    return _run(MULTI)


# ------------------------------------------------- the two layers stay apart
def test_the_two_vocabularies_are_disjoint_in_purpose():
    """Detector states are not coaching phases and must not drift into each other.

    If someone adds `loading` back to the detector, this fails and says why:
    coaching detail belongs in the analysis layer, where it costs nothing,
    not in the detector, where every extra state is another place to lose a
    shot.
    """
    assert len(CORE_STATES) == 4, CORE_STATES
    assert len(PHASE_ORDER) == 7, PHASE_ORDER
    overlap = set(CORE_STATES) & set(PHASE_ORDER)
    assert overlap <= {"release"}, (
        f"detector states leaking into the coaching vocabulary: {overlap}"
    )


def test_every_rule_targets_a_phase_the_refiner_can_emit():
    """The silent-failure guard.

    A rule keyed to a phase that no longer exists never fires. The shot still
    scores, nothing raises, and one coaching section is simply absent from the
    report. This is exactly what happened to `knee_flexion` when it was
    removed, and it is why that removal had to touch biomechanics.yaml too.
    """
    rules = load_yaml("biomechanics.yaml").get("rules", {}) or {}
    known = set(PHASE_ORDER)
    for rule_id, rule in rules.items():
        for phase in rule.get("phases", []) or []:
            assert phase in known, (
                f"rule '{rule_id}' targets phase '{phase}', which the refiner "
                f"never emits, so the rule can never fire. Known phases: "
                f"{sorted(known)}"
            )


# ------------------------------------------------------------ single attempt
def test_single_shot_produces_coaching_phases(single_shot_run):
    """Four detector states in; several coaching phases out."""
    assert single_shot_run.shots, "no shot detected in a single-attempt clip"
    shot = single_shot_run.shots[0]

    labelled = {ps.phase for ps in shot.phase_scores}
    coaching = labelled & set(PHASE_ORDER)
    assert len(coaching) >= 3, (
        f"expected the shot to be broken into coaching phases, got {labelled}"
    )


def test_single_shot_phases_are_chronological(single_shot_run):
    """Reported phases follow the order of the movement."""
    shot = single_shot_run.shots[0]
    seen = [p for p in shot.phases_seen if p in PHASE_ORDER]
    indices = [phase_index(p) for p in seen]
    assert indices == sorted(indices), seen


def test_single_shot_phase_scores_are_real(single_shot_run):
    """Each reported phase carries rules that actually ran in it.

    `whole_shot` is exempt by design: it is the one section that deliberately
    gathers rules from several phases -- posture and head stability are checked
    continuously, so printing them under every phase reads as a stuck record.
    Its members therefore carry the phase they were measured in, not the
    heading they are printed under.
    """
    shot = single_shot_run.shots[0]
    scored = [ps for ps in shot.phase_scores if ps.rules]
    assert scored, "phases were reported with no rules evaluated in any of them"
    for ps in scored:
        assert 0 <= ps.score <= 100, (ps.phase, ps.score)
        if ps.phase == WHOLE_SHOT_PHASE:
            continue
        for rule in ps.rules:
            assert rule.phase == ps.phase, (
                f"rule '{rule.rule_id}' filed under '{ps.phase}' but carries "
                f"phase '{rule.phase}'"
            )


def test_the_dip_is_analysed_even_though_no_state_detects_it(single_shot_run):
    """The point of back-fill plus post-hoc refinement.

    Nothing in the four-state detector looks for a knee dip. The loading rules
    must still fire, because the tracker back-fills the frames before the rise
    and the refiner finds the dip by its knee minimum. If this fails, the
    state machine was simplified at the cost of the coaching detail it was
    supposed to preserve.
    """
    shot = single_shot_run.shots[0]
    by_phase = {ps.phase: ps for ps in shot.phase_scores}
    assert "loading" in by_phase, (
        f"the dip was never analysed; phases present: {sorted(by_phase)}"
    )
    assert by_phase["loading"].rules, "loading phase present but scored nothing"


# --------------------------------------------------------- continuous shooting
def test_multi_shot_video_separates_every_attempt(multi_shot_run):
    """Three attempts in one recording come out as three shots.

    This is the case release-anchored segmentation exists for. Forward-opening
    segmentation returned 1-2 here, because recovery after each attempt ran
    longer than the gap before the next one.
    """
    assert len(multi_shot_run.shots) == 3, (
        f"{len(multi_shot_run.shots)} shot(s), "
        f"{multi_shot_run.discarded_candidates} discarded"
    )


def test_multi_shot_attempts_do_not_overlap(multi_shot_run):
    """Each attempt owns its own frames — no shot absorbs the next one."""
    spans = [
        (s.start_timestamp_ms, s.end_timestamp_ms)
        for s in multi_shot_run.shots
        if s.start_timestamp_ms is not None and s.end_timestamp_ms is not None
    ]
    assert spans == sorted(spans), spans
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start >= prev_end, (
            f"attempt starting at {next_start} ms overlaps the previous one, "
            f"which ran to {prev_end} ms"
        )


def test_every_attempt_in_a_multi_shot_video_is_analysed(multi_shot_run):
    """Being found is not enough; each attempt must produce real coaching."""
    for shot in multi_shot_run.shots:
        if shot.is_rejected:
            continue
        coaching = {ps.phase for ps in shot.phase_scores} & set(PHASE_ORDER)
        assert coaching, f"shot {shot.shot_number} produced no phase analysis"
        assert shot.score is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
