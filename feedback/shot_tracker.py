"""
Detect shot boundaries and finalize scoring when a rep completes.

WHY THIS IS CANDIDATE-BASED
---------------------------
The previous implementation opened a new shot on any
``ready_stance -> loading|ball_lift`` transition. That is not a shooting
event, it is a posture change, so every re-settle, stance adjustment or
bounce on the toes produced another "shot". Measured before the fix: one
45 s clip of roughly one shooting sequence reported 7 shots, and a 15 s
free-throw clip reported 4.

Now a transition only opens a CANDIDATE. A candidate becomes a confirmed
shot only if it shows a credible shooting event -- the ball is released, and
the shooting wrist actually travelled the distance a shot requires. A
candidate that never produces a release is discarded silently. That is what
makes toe-jiggling produce one shot instead of three, and it generalises,
because it tests the movement rather than the fixture.

All timing is in SECONDS, derived from frame timestamps. Sample material
spans 12-30 fps and includes slow motion, so frame counts are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ball.models import ShotOutcome
from feedback.generator import generate_coaching_tips
from feedback.models import ShotSummary
from feedback.performance_plan import ACTIVE_SHOT_PHASES, build_shot_performance_plan
from feedback.scorer import score_shot
from phase_detection.phases import PHASE_LABELS
from shots.classifier import AttemptEvidence, classify
from shots.types import RejectionReason, ShotType
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameSnapshot


@dataclass
class _Candidate:
    """A possible shot, accumulating evidence until confirmed or discarded."""

    started_ms: int
    entry_phase: str
    mid_start: bool
    frames: List[FrameSnapshot] = field(default_factory=list)
    reached_release: bool = False
    body_finished: bool = False
    body_finished_ms: Optional[int] = None

    # Evidence
    wrist_min_y: float = float("inf")
    wrist_max_y: float = float("-inf")
    wrist_above_shoulder_max: float = float("-inf")
    vertical_displacement_max: float = 0.0
    hip_x_min: float = float("inf")
    hip_x_max: float = float("-inf")
    hip_z_min: float = float("inf")
    hip_z_max: float = float("-inf")
    last_ms: int = 0

    def observe(self, snapshot: FrameSnapshot) -> None:
        self.frames.append(snapshot)
        self.last_ms = snapshot.timestamp_ms
        f = snapshot.features
        if f is None:
            return

        self.wrist_min_y = min(self.wrist_min_y, f.wrist_y)
        self.wrist_max_y = max(self.wrist_max_y, f.wrist_y)
        self.wrist_above_shoulder_max = max(
            self.wrist_above_shoulder_max, f.wrist_y - f.shoulder_y
        )
        self.hip_x_min = min(self.hip_x_min, f.hip_x_ratio)
        self.hip_x_max = max(self.hip_x_max, f.hip_x_ratio)

        # Vertical displacement is measured only from ball lift onward, so a
        # bounce while settling in ready_stance cannot masquerade as a jump.
        # body_rise_ratio comes from IMAGE space because world landmarks are
        # hip-centred and therefore blind to whole-body translation.
        if snapshot.phase in ("ball_lift", "jump", "release"):
            self.vertical_displacement_max = max(
                self.vertical_displacement_max, f.body_rise_ratio
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.last_ms - self.started_ms) / 1000.0)

    @property
    def wrist_rise_m(self) -> float:
        if self.wrist_max_y == float("-inf") or self.wrist_min_y == float("inf"):
            return 0.0
        return self.wrist_max_y - self.wrist_min_y

    @property
    def horizontal_travel_m(self) -> float:
        """Horizontal hip travel across the frame, in image-normalised units."""
        if self.hip_x_max == float("-inf"):
            return 0.0
        return float(self.hip_x_max - self.hip_x_min)

    def evidence(self) -> AttemptEvidence:
        return AttemptEvidence(
            vertical_displacement_m=self.vertical_displacement_max,
            horizontal_travel_m=self.horizontal_travel_m,
            wrist_rise_m=self.wrist_rise_m,
            wrist_above_shoulder_m=(
                self.wrist_above_shoulder_max
                if self.wrist_above_shoulder_max != float("-inf")
                else -1.0
            ),
            reached_release=self.reached_release,
            duration_s=self.duration_s,
        )


class ShotTracker:
    """Coordinate body-form capture with the terminal ball outcome."""

    def __init__(self):
        cfg = load_yaml("segmentation.yaml")
        confirm = cfg.get("confirmation", {}) or {}
        self._require_release = bool(confirm.get("require_release", True))
        self._min_wrist_rise_m = float(confirm.get("min_wrist_rise_m", 0.18))
        self._min_wrist_above_shoulder_m = float(
            confirm.get("min_wrist_above_shoulder_m", -0.12)
        )
        self._min_duration_s = float(confirm.get("min_duration_s", 0.20))
        self._max_duration_s = float(confirm.get("max_duration_s", 12.0))
        self._refractory_s = float(cfg.get("refractory_s", 0.60))
        self._candidate_timeout_s = float(cfg.get("candidate_timeout_s", 6.0))

        self.shot_count = 0
        self.last_summary: Optional[ShotSummary] = None
        self.last_score: Optional[int] = None
        self.discarded_candidates = 0

        self._prev_phase = "ready_stance"
        self._candidate: Optional[_Candidate] = None
        self._summary_display_frames = 0
        self._first_shot_pending = True
        self._last_confirmed_ms: Optional[int] = None

        self._require_ball_outcome = False
        self._body_grace_ms = 500
        self._ball_outcome: Optional[ShotOutcome] = None
        self._ball_outcome_timestamp_ms: Optional[int] = None

    # ------------------------------------------------------------------ api
    @property
    def shot_in_progress(self) -> bool:
        return self._candidate is not None

    @property
    def capture_in_progress(self) -> bool:
        return self._candidate is not None and not self._candidate.body_finished

    @property
    def show_shot_summary(self) -> bool:
        return self._summary_display_frames > 0 and self.last_summary is not None

    @property
    def capture_warning(self) -> Optional[str]:
        c = self._candidate
        if c is not None and c.mid_start:
            label = PHASE_LABELS.get(c.entry_phase, c.entry_phase)
            return f"Recording started at {label} — earlier phases not on camera"
        return None

    def configure_ball_outcome(self, required: bool, body_grace_ms: int = 500) -> None:
        self._require_ball_outcome = bool(required)
        self._body_grace_ms = max(0, int(body_grace_ms))

    def begin_summary_display(self, frames: int) -> None:
        self._summary_display_frames = frames

    def reset(self) -> None:
        required, grace = self._require_ball_outcome, self._body_grace_ms
        self.__init__()
        self.configure_ball_outcome(required, grace)

    # -------------------------------------------------------------- updates
    def update(
        self,
        phase: str,
        snapshot: FrameSnapshot,
        ball_outcome: Optional[ShotOutcome] = None,
    ) -> Optional[ShotSummary]:
        now_ms = snapshot.timestamp_ms
        completed: Optional[ShotSummary] = None

        if self._candidate is None:
            if self._should_open(phase, now_ms):
                mid = self._first_shot_pending and not self._is_start_transition(phase)
                self._open(phase, now_ms, mid_start=mid)
        else:
            if phase == "release":
                self._candidate.reached_release = True

        if self._candidate is not None and not self._candidate.body_finished:
            self._candidate.observe(snapshot)
            if self._prev_phase == "landing" and phase == "ready_stance":
                self._candidate.body_finished = True
                self._candidate.body_finished_ms = now_ms

        self._accept_ball_outcome(ball_outcome, now_ms)

        if self._candidate is not None:
            if self._candidate.duration_s > self._candidate_timeout_s and not (
                self._candidate.reached_release
            ):
                # No release ever arrived. This is the toe-jiggle path.
                self._discard()
            elif self._ready_to_finalize(now_ms):
                completed = self._finalize(now_ms)

        self._prev_phase = phase
        self._advance_summary_display(completed)
        return completed

    def update_ball_outcome(
        self,
        outcome: Optional[ShotOutcome],
        timestamp_ms: int,
    ) -> Optional[ShotSummary]:
        """Continue/finalize an active attempt when pose is absent this frame."""
        if self._candidate is None:
            self._advance_summary_display(None)
            return None

        self._accept_ball_outcome(outcome, timestamp_ms)
        completed = None
        if self._ready_to_finalize(timestamp_ms):
            completed = self._finalize(timestamp_ms)
        self._advance_summary_display(completed)
        return completed

    def finalize_in_progress(self, min_frames: int = 12) -> Optional[ShotSummary]:
        """Close an attempt still open when the video or session ends."""
        c = self._candidate
        if c is None or len(c.frames) < min_frames:
            self._candidate = None
            return None
        if self._require_ball_outcome and self._ball_outcome is None:
            self._ball_outcome = self._unknown_outcome(
                c.last_ms, "Session ended before ball outcome was resolved"
            )
        summary = self._finalize(c.last_ms, force=True)
        if summary is not None:
            self._summary_display_frames = 90
        return summary

    # ----------------------------------------------------------- internals
    def _is_start_transition(self, phase: str) -> bool:
        return self._prev_phase == "ready_stance" and phase in ("loading", "ball_lift")

    def _should_open(self, phase: str, now_ms: int) -> bool:
        if self._last_confirmed_ms is not None:
            if (now_ms - self._last_confirmed_ms) / 1000.0 < self._refractory_s:
                return False
        if self._is_start_transition(phase):
            return True
        # Allow a first shot already underway when recording began.
        return self._first_shot_pending and phase in ACTIVE_SHOT_PHASES

    def _open(self, phase: str, now_ms: int, mid_start: bool) -> None:
        self._candidate = _Candidate(
            started_ms=now_ms, entry_phase=phase, mid_start=mid_start
        )
        self._first_shot_pending = False
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None

    def _discard(self) -> None:
        self.discarded_candidates += 1
        self._candidate = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None

    def _is_credible(self, c: _Candidate) -> bool:
        """Does this candidate show a real shooting event?"""
        if self._require_release and not c.reached_release:
            return False
        if c.duration_s < self._min_duration_s:
            return False
        if c.duration_s > self._max_duration_s:
            return False
        if c.wrist_rise_m < self._min_wrist_rise_m:
            return False
        if c.wrist_above_shoulder_max < self._min_wrist_above_shoulder_m:
            return False
        return True

    def _accept_ball_outcome(self, outcome, timestamp_ms: int) -> None:
        if self._candidate is None or outcome is None or self._ball_outcome is not None:
            return
        self._ball_outcome = outcome
        self._ball_outcome_timestamp_ms = (
            outcome.outcome_timestamp_ms
            if outcome.outcome_timestamp_ms is not None
            else timestamp_ms
        )

    def _ready_to_finalize(self, timestamp_ms: int) -> bool:
        c = self._candidate
        if c is None:
            return False
        if not self._require_ball_outcome:
            return c.body_finished
        if self._ball_outcome is None:
            return False
        if c.body_finished:
            return True
        if self._ball_outcome_timestamp_ms is None:
            return False
        return timestamp_ms - self._ball_outcome_timestamp_ms >= self._body_grace_ms

    def _finalize(self, now_ms: int, force: bool = False) -> Optional[ShotSummary]:
        c = self._candidate
        if c is None:
            return None

        # An attempt can be finalised by the ball outcome on a frame where no
        # pose was observed, so the candidate's own clock may lag. The attempt
        # genuinely spans up to this moment.
        c.last_ms = max(c.last_ms, now_ms)

        if not force and not self._is_credible(c):
            self._discard()
            return None
        if force and not c.reached_release:
            self._discard()
            return None

        classification = classify(c.evidence())

        self.shot_count += 1
        if not classification.is_implemented:
            # Recognised, but we have no model for it. Produce a rejection,
            # never a score computed with the wrong biomechanics.
            summary = ShotSummary(
                shot_number=self.shot_count,
                score=None,
                passed_count=0,
                total_count=0,
                phases_seen=self._phases_seen(c),
                entry_phase=c.entry_phase,
                shot_type=classification.shot_type,
                classification=classification,
                rejection=classification.rejection
                or RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET,
            )
        else:
            summary = score_shot(
                c.frames,
                self.shot_count,
                started_mid_phase=c.mid_start,
                ended_early=not c.body_finished,
                entry_phase=c.entry_phase,
            )
            summary.shot_type = classification.shot_type
            summary.classification = classification
            summary.outcome = self._ball_outcome
            summary.coaching_tips = generate_coaching_tips(summary)
            summary = build_shot_performance_plan(summary)

        self.last_summary = summary
        self.last_score = summary.score
        self._last_confirmed_ms = now_ms
        self._candidate = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None
        return summary

    @staticmethod
    def _phases_seen(c: _Candidate) -> List[str]:
        seen: List[str] = []
        for f in c.frames:
            if f.phase not in seen:
                seen.append(f.phase)
        return seen

    @staticmethod
    def _unknown_outcome(timestamp_ms: int, evidence: str) -> ShotOutcome:
        return ShotOutcome(
            result="unknown",
            confidence=0.0,
            outcome_timestamp_ms=timestamp_ms,
            evidence=[evidence],
        )

    def _advance_summary_display(self, completed: Optional[ShotSummary]) -> None:
        if completed is not None:
            self._summary_display_frames = 90
        elif self._summary_display_frames > 0:
            self._summary_display_frames -= 1
