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

    def __init__(self, config_path: str = "config/ball.yaml"):
        """Initialize outcome fusion with all sub-modules."""
        self.config = self._load_config(config_path)
        self._release_sync = ReleaseSync(config_path)
        self._trajectory = TrajectoryAnalyzer(config_path)
        self._outcome = OutcomeClassifier(config_path)
        self.post_shot_capture_ms = self.config.get("post_shot_capture_ms", 500)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    def fuse(
        self,
        shot_summary: ShotSummary,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int] = None,
    ) -> ShotSummary:
        """Return ShotSummary extended with outcome field."""
        if shot_summary is None:
            return shot_summary

        # Get shot time window
        start_ms = shot_summary.start_time_ms
        end_ms = shot_summary.end_time_ms
        
        # Extend end time to capture post-shot
        end_ms_extended = end_ms + self.post_shot_capture_ms

        # Compute outcome for the shot window
        outcome = self.finalize_shot_outcome(
            ball_buffer,
            start_ms,
            end_ms_extended
        )

        # Attach outcome to shot summary
        shot_summary.outcome = outcome
        
        # Add coaching note if outcome provides insights
        if outcome and outcome.confidence > 0.6:
            if outcome.outcome == "missed":
                shot_summary.coaching_note = self._generate_coaching_note(
                    outcome, ball_buffer, body_release_frame
                )

        return shot_summary

    def finalize_shot_outcome(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        start_ms: int,
        end_ms: int,
    ) -> Optional[ShotOutcome]:
        """Compute outcome for one shot window without form summary."""
        if not ball_buffer or len(ball_buffer) == 0:
            return ShotOutcome(
                outcome="unknown",
                confidence=0.0,
                evidence="No ball data available"
            )

        # Slice buffer to shot window + post-shot capture
        window_snapshots = ball_buffer.get_window(start_ms, end_ms)
        
        if len(window_snapshots) < 3:
            return ShotOutcome(
                outcome="unknown",
                confidence=0.0,
                evidence="Insufficient ball data in window"
            )

        # Create temporary buffer with window data
        temp_buffer = BallTimeSeriesBuffer()
        for snapshot in window_snapshots:
            temp_buffer.push(snapshot)

        # Analyze trajectory
        trajectory = self._trajectory.analyze_shot_window(temp_buffer)

        # Classify outcome
        outcome = self._outcome.classify(temp_buffer, trajectory)

        return outcome

    def _generate_coaching_note(
        self,
        outcome: ShotOutcome,
        ball_buffer: BallTimeSeriesBuffer,
        body_release_frame: Optional[int]
    ) -> str:
        """Generate coaching note based on outcome and evidence."""
        notes = []
        
        if outcome.evidence:
            notes.append(f"Evidence: {outcome.evidence}")
        
        # Check for release timing issues
        if body_release_frame is not None:
            events = ball_buffer.detect_events()
            ball_release = events.get('release')
            
            if ball_release is not None:
                offset = abs(ball_release - body_release_frame)
                if offset > 5:
                    notes.append(f"Release timing offset: {offset} frames")
                    if outcome.outcome == "missed":
                        notes.append("Consider adjusting release timing")

        # Check trajectory quality
        if hasattr(outcome, 'trajectory') and outcome.trajectory:
            if outcome.trajectory.r_squared < 0.7:
                notes.append("Inconsistent ball flight path")
                if outcome.outcome == "missed":
                    notes.append("Work on consistent release mechanics")

        return "; ".join(notes) if notes else "No specific coaching note"