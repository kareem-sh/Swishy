"""Detect shot boundaries and finalize scoring when a rep completes."""

from __future__ import annotations

from typing import List, Optional

from ball.models import ShotOutcome
from feedback.generator import generate_coaching_tips
from feedback.models import ShotSummary
from feedback.performance_plan import ACTIVE_SHOT_PHASES, build_shot_performance_plan
from feedback.scorer import score_shot
from phase_detection.phases import PHASE_LABELS
from utils.frame_buffer import FrameSnapshot


class ShotTracker:
    """Coordinate body-form capture with the terminal ball outcome.

    Body scoring stops at landing -> ready stance.  When ball outcome tracking
    is required, the logical attempt remains open until made/missed/unknown is
    available.  Conversely, if the ball finishes first, a short grace period
    lets the pose detector capture follow-through and landing.
    """

    def __init__(self):
        self.shot_in_progress = False
        self.shot_count = 0
        self.last_summary: Optional[ShotSummary] = None
        self.last_score: Optional[int] = None
        self._prev_phase = "ready_stance"
        self._shot_frames: List[FrameSnapshot] = []
        self._summary_display_frames = 0
        self._first_shot_pending = True
        self._started_mid_phase = False
        self._ended_early = False
        self._entry_phase: Optional[str] = None

        self._require_ball_outcome = False
        self._body_grace_ms = 500
        self._body_finished = False
        self._body_finished_timestamp_ms: Optional[int] = None
        self._ball_outcome: Optional[ShotOutcome] = None
        self._ball_outcome_timestamp_ms: Optional[int] = None

    def configure_ball_outcome(self, required: bool, body_grace_ms: int = 500) -> None:
        """Configure whether a terminal ball result is required per attempt."""
        self._require_ball_outcome = bool(required)
        self._body_grace_ms = max(0, int(body_grace_ms))

    def update(
        self,
        phase: str,
        snapshot: FrameSnapshot,
        ball_outcome: Optional[ShotOutcome] = None,
    ) -> Optional[ShotSummary]:
        """Process body and ball state; return a newly completed shot if any."""
        completed: Optional[ShotSummary] = None
        starts_new_shot = self._is_start_transition(phase)

        # A new loading motion must not be swallowed by an unresolved previous
        # attempt. Close the old one as unknown, then begin the new rep.
        if self.shot_in_progress and self._body_finished and starts_new_shot:
            if self._ball_outcome is None and self._require_ball_outcome:
                self._ball_outcome = self._unknown_outcome(
                    snapshot.timestamp_ms,
                    "New shot began before previous ball outcome was resolved",
                )
            completed = self._finish_current(ended_early=False)
            self._begin_shot(phase, mid_start=False)

        elif not self.shot_in_progress:
            if starts_new_shot:
                self._begin_shot(phase, mid_start=False)
            elif self._first_shot_pending and phase in ACTIVE_SHOT_PHASES:
                mid = (
                    phase not in ("loading", "ball_lift")
                    or self._prev_phase != "ready_stance"
                )
                self._begin_shot(phase, mid_start=mid)

        if self.shot_in_progress and not self._body_finished:
            self._shot_frames.append(snapshot)

        if (
            self.shot_in_progress
            and not self._body_finished
            and self._prev_phase == "landing"
            and phase == "ready_stance"
        ):
            self._body_finished = True
            self._body_finished_timestamp_ms = snapshot.timestamp_ms

        self._accept_ball_outcome(ball_outcome, snapshot.timestamp_ms)

        if (
            completed is None
            and self.shot_in_progress
            and self._ready_to_finalize(snapshot.timestamp_ms)
        ):
            completed = self._finish_current(ended_early=not self._body_finished)

        self._prev_phase = phase
        self._advance_summary_display(completed)
        return completed

    def update_ball_outcome(
        self,
        outcome: Optional[ShotOutcome],
        timestamp_ms: int,
    ) -> Optional[ShotSummary]:
        """Continue/finalize an active attempt when pose is absent this frame."""
        if not self.shot_in_progress:
            self._advance_summary_display(None)
            return None

        self._accept_ball_outcome(outcome, timestamp_ms)
        completed = None
        if self._ready_to_finalize(timestamp_ms):
            completed = self._finish_current(ended_early=not self._body_finished)
        self._advance_summary_display(completed)
        return completed

    def _is_start_transition(self, phase: str) -> bool:
        return self._prev_phase == "ready_stance" and phase in (
            "loading",
            "ball_lift",
        )

    def _begin_shot(self, phase: str, mid_start: bool) -> None:
        self.shot_in_progress = True
        self._shot_frames = []
        self._first_shot_pending = False
        self._started_mid_phase = mid_start
        self._ended_early = False
        self._entry_phase = phase
        self._body_finished = False
        self._body_finished_timestamp_ms = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None

    def _accept_ball_outcome(
        self,
        outcome: Optional[ShotOutcome],
        timestamp_ms: int,
    ) -> None:
        if not self.shot_in_progress or outcome is None or self._ball_outcome is not None:
            return
        self._ball_outcome = outcome
        self._ball_outcome_timestamp_ms = (
            outcome.outcome_timestamp_ms
            if outcome.outcome_timestamp_ms is not None
            else timestamp_ms
        )

    def _ready_to_finalize(self, timestamp_ms: int) -> bool:
        if not self.shot_in_progress:
            return False
        if not self._require_ball_outcome:
            return self._body_finished
        if self._ball_outcome is None:
            return False
        if self._body_finished:
            return True
        if self._ball_outcome_timestamp_ms is None:
            return False
        return timestamp_ms - self._ball_outcome_timestamp_ms >= self._body_grace_ms

    def _finish_current(self, ended_early: bool) -> ShotSummary:
        summary = self._finalize(ended_early=ended_early)
        self.shot_in_progress = False
        self._shot_frames = []
        self._started_mid_phase = False
        self._ended_early = False
        self._entry_phase = None
        self._body_finished = False
        self._body_finished_timestamp_ms = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None
        return summary

    def _finalize(self, ended_early: bool) -> ShotSummary:
        self.shot_count += 1
        summary = score_shot(
            self._shot_frames,
            self.shot_count,
            started_mid_phase=self._started_mid_phase,
            ended_early=ended_early,
            entry_phase=self._entry_phase,
        )
        summary.outcome = self._ball_outcome
        summary.coaching_tips = generate_coaching_tips(summary)
        summary = build_shot_performance_plan(summary)
        self.last_summary = summary
        self.last_score = summary.score
        return summary

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

    @property
    def capture_in_progress(self) -> bool:
        """Whether body frames should still be recorded for form scoring."""
        return self.shot_in_progress and not self._body_finished

    @property
    def show_shot_summary(self) -> bool:
        return self._summary_display_frames > 0 and self.last_summary is not None

    @property
    def capture_warning(self) -> str | None:
        if self.shot_in_progress and self._started_mid_phase and self._entry_phase:
            label = PHASE_LABELS.get(self._entry_phase, self._entry_phase)
            return f"Recording started at {label} — earlier phases not on camera"
        return None

    def begin_summary_display(self, frames: int) -> None:
        self._summary_display_frames = frames

    def finalize_in_progress(self, min_frames: int = 12) -> Optional[ShotSummary]:
        """Close a shot still active when a video/live session ends."""
        if not self.shot_in_progress or len(self._shot_frames) < min_frames:
            return None
        if self._require_ball_outcome and self._ball_outcome is None:
            timestamp_ms = (
                self._shot_frames[-1].timestamp_ms if self._shot_frames else 0
            )
            self._ball_outcome = self._unknown_outcome(
                timestamp_ms,
                "Session ended before ball outcome was resolved",
            )
        summary = self._finish_current(ended_early=not self._body_finished)
        self._summary_display_frames = 90
        return summary

    def reset(self) -> None:
        required = self._require_ball_outcome
        grace_ms = self._body_grace_ms
        self.__init__()
        self.configure_ball_outcome(required, grace_ms)
