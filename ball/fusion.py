"""Fuse form analysis with ball outcome (Phase 6)."""

from __future__ import annotations

from typing import Optional

from ball.models import ShotOutcome
from ball.outcome import OutcomeClassifier
from ball.release_sync import ReleaseSync
from ball.timeseries import BallTimeSeriesBuffer
from ball.trajectory import TrajectoryAnalyzer
from feedback.models import ShotSummary
from utils.config_loader import load_yaml


class OutcomeFusion:
    """Attach ShotOutcome to completed ShotSummary."""

    def __init__(self, config_name: str = "ball.yaml"):
        self.config = load_yaml(config_name)
        self._release_sync = ReleaseSync(config_name)
        self._trajectory = TrajectoryAnalyzer(config_name)
        self._outcome = OutcomeClassifier(config_name)
        outcome_cfg = self.config.get("outcome", {})
        self.post_shot_capture_ms = int(
            outcome_cfg.get(
                "post_shot_capture_ms",
                self.config.get("post_shot_capture_ms", 2000),
            )
        )

    def fuse(
        self,
        shot_summary: ShotSummary,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int] = None,
        start_ms: int = 0,
        end_ms: int = 0,
    ) -> ShotSummary:
        """Return ShotSummary extended with outcome field when supported."""
        if shot_summary is None:
            return shot_summary

        if end_ms <= start_ms:
            end_ms = start_ms + self.post_shot_capture_ms
        else:
            end_ms = end_ms + self.post_shot_capture_ms

        outcome = self.finalize_shot_outcome(ball_buffer, start_ms, end_ms)
        if hasattr(shot_summary, "outcome"):
            shot_summary.outcome = outcome
        return shot_summary

    def finalize_shot_outcome(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        start_ms: int,
        end_ms: int,
    ) -> Optional[ShotOutcome]:
        if not ball_buffer or len(ball_buffer) == 0:
            return ShotOutcome(
                result="unknown",
                confidence=0.0,
                evidence=["No ball data available"],
            )

        window_snapshots = ball_buffer.get_window(start_ms, end_ms)
        if len(window_snapshots) < 3:
            return ShotOutcome(
                result="unknown",
                confidence=0.0,
                evidence=["Insufficient ball data in window"],
            )

        temp_buffer = BallTimeSeriesBuffer()
        for snapshot in window_snapshots:
            temp_buffer.push(snapshot)

        trajectory = self._trajectory.analyze_shot_window(temp_buffer)
        return self._outcome.classify(temp_buffer, trajectory)
