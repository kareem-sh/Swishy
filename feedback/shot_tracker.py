"""Detect shot boundaries and finalize scoring when a rep completes."""

from typing import List, Optional

from feedback.generator import generate_coaching_tips
from feedback.models import ShotSummary
from feedback.scorer import score_shot
from utils.frame_buffer import FrameSnapshot


class ShotTracker:
    """
    Tracks one jump-shot attempt from loading through landing.

    A shot starts: ready_stance -> loading
    A shot ends:   landing -> ready_stance
    """

    def __init__(self):
        self.shot_in_progress = False
        self.shot_count = 0
        self.last_summary: Optional[ShotSummary] = None
        self.last_score: Optional[int] = None
        self._prev_phase = "ready_stance"
        self._shot_frames: List[FrameSnapshot] = []
        self._summary_display_frames = 0

    def update(self, phase: str, snapshot: FrameSnapshot) -> Optional[ShotSummary]:
        """Process phase change; return ShotSummary when a shot just completed."""
        completed: Optional[ShotSummary] = None

        if not self.shot_in_progress and self._prev_phase == "ready_stance" and phase in ("loading", "ball_lift"):
            self.shot_in_progress = True
            self._shot_frames = []

        if self.shot_in_progress:
            self._shot_frames.append(snapshot)

        if self.shot_in_progress and self._prev_phase == "landing" and phase == "ready_stance":
            completed = self._finalize()
            self.shot_in_progress = False
            self._shot_frames = []

        self._prev_phase = phase

        if completed is not None:
            self._summary_display_frames = 90
        elif self._summary_display_frames > 0:
            self._summary_display_frames -= 1

        return completed

    def _finalize(self) -> ShotSummary:
        self.shot_count += 1
        summary = score_shot(self._shot_frames, self.shot_count)
        summary.coaching_tips = generate_coaching_tips(summary)
        self.last_summary = summary
        self.last_score = summary.score
        return summary

    @property
    def show_shot_summary(self) -> bool:
        return self._summary_display_frames > 0 and self.last_summary is not None

    def begin_summary_display(self, frames: int):
        self._summary_display_frames = frames

    def finalize_in_progress(self, min_frames: int = 12) -> Optional[ShotSummary]:
        """Score and close a shot that was still in progress when the session ended."""
        if not self.shot_in_progress or len(self._shot_frames) < min_frames:
            return None
        self.shot_in_progress = False
        summary = self._finalize()
        self._shot_frames = []
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
