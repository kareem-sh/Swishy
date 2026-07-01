"""Fuse form analysis with ball outcome (Phase 6)."""

from typing import Optional

from ball.models import ShotOutcome
from ball.outcome import OutcomeClassifier
from ball.release_sync import ReleaseSync
from ball.timeseries import BallTimeSeriesBuffer
from ball.trajectory import TrajectoryAnalyzer
from feedback.models import ShotSummary


class OutcomeFusion:
    """Attach ShotOutcome to completed ShotSummary."""

    def __init__(self):
        self._release_sync = ReleaseSync()
        self._trajectory = TrajectoryAnalyzer()
        self._outcome = OutcomeClassifier()
        # TODO: wire sub-modules when implemented

    def fuse(
        self,
        shot_summary: ShotSummary,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int] = None,
    ) -> ShotSummary:
        """Return ShotSummary extended with outcome field."""
        # TODO: run release sync → trajectory → outcome pipeline
        # TODO: attach ShotOutcome to shot_summary.outcome (extend model)
        # TODO: add form–outcome coaching note when useful
        pass

    def finalize_shot_outcome(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        start_ms: int,
        end_ms: int,
    ) -> Optional[ShotOutcome]:
        """Compute outcome for one shot window without form summary."""
        # TODO: slice buffer to shot window + post_shot_capture_ms
        pass
