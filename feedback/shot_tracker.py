"""Detect shot boundaries and finalize scoring when a rep completes."""

from typing import List, Optional

from feedback.generator import generate_coaching_tips
from feedback.models import ShotSummary
from feedback.performance_plan import ACTIVE_SHOT_PHASES, build_shot_performance_plan
from feedback.scorer import score_shot
from phase_detection.phases import PHASE_LABELS
from utils.frame_buffer import FrameSnapshot


class ShotTracker:
    """
    Tracks one jump-shot attempt from loading through landing.

    Normal start: ready_stance -> loading / ball_lift
    Mid-entry start: first frames of a session already inside a shot phase
    (e.g. video starts at release) — still tracks through landing.
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

    def update(self, phase: str, snapshot: FrameSnapshot) -> Optional[ShotSummary]:
        """Process phase change; return ShotSummary when a shot just completed."""
        completed: Optional[ShotSummary] = None

        if not self.shot_in_progress:
            if self._prev_phase == "ready_stance" and phase in ("loading", "ball_lift"):
                self._begin_shot(phase, mid_start=False)
            elif self._first_shot_pending and phase in ACTIVE_SHOT_PHASES:
                mid = phase not in ("loading", "ball_lift") or self._prev_phase != "ready_stance"
                self._begin_shot(phase, mid_start=mid)

        if self.shot_in_progress:
            self._shot_frames.append(snapshot)

        if self.shot_in_progress and self._prev_phase == "landing" and phase == "ready_stance":
            completed = self._finalize(ended_early=False)
            self.shot_in_progress = False
            self._shot_frames = []
            self._started_mid_phase = False
            self._ended_early = False
            self._entry_phase = None

        self._prev_phase = phase

        if completed is not None:
            self._summary_display_frames = 90
        elif self._summary_display_frames > 0:
            self._summary_display_frames -= 1

        return completed

    def _begin_shot(self, phase: str, mid_start: bool):
        self.shot_in_progress = True
        self._shot_frames = []
        self._first_shot_pending = False
        self._started_mid_phase = mid_start
        self._ended_early = False
        self._entry_phase = phase

    def _finalize(self, ended_early: bool) -> ShotSummary:
        self.shot_count += 1
        summary = score_shot(
            self._shot_frames,
            self.shot_count,
            started_mid_phase=self._started_mid_phase,
            ended_early=ended_early,
            entry_phase=self._entry_phase,
        )
        summary.coaching_tips = generate_coaching_tips(summary)
        summary = build_shot_performance_plan(summary)
        self.last_summary = summary
        self.last_score = summary.score
        return summary

    @property
    def show_shot_summary(self) -> bool:
        return self._summary_display_frames > 0 and self.last_summary is not None

    @property
    def capture_warning(self) -> str | None:
        if self.shot_in_progress and self._started_mid_phase and self._entry_phase:
            label = PHASE_LABELS.get(self._entry_phase, self._entry_phase)
            return f"Recording started at {label} — earlier phases not on camera"
        return None

    def begin_summary_display(self, frames: int):
        self._summary_display_frames = frames

    def finalize_in_progress(self, min_frames: int = 12) -> Optional[ShotSummary]:
        """Score and close a shot that was still in progress when the session ended."""
        if not self.shot_in_progress or len(self._shot_frames) < min_frames:
            return None
        self.shot_in_progress = False
        summary = self._finalize(ended_early=True)
        self._shot_frames = []
        self._started_mid_phase = False
        self._ended_early = False
        self._entry_phase = None
        self._summary_display_frames = 90
        return summary

    def reset(self):
        self.shot_in_progress = False
        self.shot_count = 0
        self.last_summary = None
        self.last_score = None
        self._prev_phase = "ready_stance"
        self._shot_frames = []
        self._summary_display_frames = 0
        self._first_shot_pending = True
        self._started_mid_phase = False
        self._ended_early = False
        self._entry_phase = None
