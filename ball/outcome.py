"""Make / miss / unknown classification (Phase 6c)."""

from typing import Optional
import numpy as np
import yaml

from ball.models import BallTrajectory, ShotOutcome
from ball.timeseries import BallTimeSeriesBuffer


class OutcomeClassifier:
    """Rule-based shot outcome from ball time-series and hoop ROI."""

    def __init__(self, config_path: str = "config/ball.yaml"):
        """Initialize outcome classifier with configuration."""
        self.config = self._load_config(config_path)
        self.hoop_roi = None
        self.rim_radius_px = self.config.get("rim_radius_px", 30)
        self.max_rim_offset = self.config.get("max_rim_offset", 20)
        self.confidence_threshold = self.config.get("outcome_confidence_threshold", 0.6)
        
        # Load hoop ROI
        self.load_hoop_roi("config/hoop_roi.yaml")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}

    def classify(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        trajectory: Optional[BallTrajectory] = None,
    ) -> ShotOutcome:
        """Return made | missed | unknown with confidence and evidence."""
        if self.hoop_roi is None:
            return ShotOutcome(
                outcome="unknown",
                confidence=0.0,
                evidence="Hoop ROI not configured"
            )

        if len(ball_buffer) < 5:
            return ShotOutcome(
                outcome="unknown",
                confidence=0.0,
                evidence="Insufficient ball data"
            )

        # Get snapshots near rim
        rim_snapshots = self._get_rim_snapshots(ball_buffer)
        
        if not rim_snapshots:
            return ShotOutcome(
                outcome="unknown",
                confidence=0.0,
                evidence="Ball did not approach rim"
            )

        # Analyze rim approach
        outcome, confidence, evidence = self._analyze_rim_approach(rim_snapshots)
        
        # Refine with trajectory if available
        if trajectory is not None:
            outcome, confidence, evidence = self._refine_with_trajectory(
                outcome, confidence, evidence, trajectory
            )

        return ShotOutcome(
            outcome=outcome,
            confidence=confidence,
            evidence=evidence
        )

    def _get_rim_snapshots(self, buffer: BallTimeSeriesBuffer) -> list:
        """Get snapshots near rim."""
        if not buffer.buffer:
            return []

        hoop_center_x = self.hoop_roi.get('center_x', 0)
        hoop_center_y = self.hoop_roi.get('center_y', 0)
        rim_radius = self.hoop_roi.get('rim_radius', self.rim_radius_px)

        rim_snapshots = []
        for snapshot in buffer.buffer:
            dist = np.sqrt((snapshot.x - hoop_center_x)**2 + (snapshot.y - hoop_center_y)**2)
            if dist <= rim_radius * 1.5:  # Within 1.5x rim radius
                rim_snapshots.append((snapshot, dist))

        return rim_snapshots

    def _analyze_rim_approach(self, rim_snapshots: list) -> tuple:
        """Analyze how ball approaches rim."""
        if not rim_snapshots:
            return "unknown", 0.0, "No snapshots near rim"

        hoop_center_x = self.hoop_roi.get('center_x', 0)
        hoop_center_y = self.hoop_roi.get('center_y', 0)
        rim_radius = self.hoop_roi.get('rim_radius', self.rim_radius_px)

        # Check if any snapshot is inside rim
        inside_rim = False
        close_to_rim = False
        
        for snapshot, dist in rim_snapshots:
            if dist < rim_radius:
                inside_rim = True
                break
            elif dist < rim_radius * 1.2:
                close_to_rim = True

        # Check trajectory direction near rim
        if len(rim_snapshots) >= 2:
            # Check if ball is moving downward (towards rim)
            y_vel = rim_snapshots[-1][0].y - rim_snapshots[0][0].y
            if y_vel > 0:  # Moving downward
                if inside_rim:
                    return "made", 0.85, "Ball entered rim area"
                elif close_to_rim:
                    # Check rim offset
                    last_snap, dist = rim_snapshots[-1]
                    offset = dist - rim_radius
                    if offset <= self.max_rim_offset:
                        return "missed", 0.65, "Ball close to rim but outside"
                    else:
                        return "missed", 0.75, "Ball significantly outside rim"
            
        return "unknown", 0.3, "Insufficient evidence"

    def _refine_with_trajectory(self, outcome: str, confidence: float, 
                               evidence: str, trajectory: BallTrajectory) -> tuple:
        """Refine outcome based on trajectory analysis."""
        if trajectory is None or trajectory.fit_params is None:
            return outcome, confidence, evidence

        # Use entry angle to refine
        entry_angle = trajectory.estimate_entry_angle() if hasattr(trajectory, 'estimate_entry_angle') else None
        
        if entry_angle is not None:
            # Optimal entry angles are around 45-52 degrees
            if 40 <= entry_angle <= 55:
                if outcome == "made":
                    confidence = min(1.0, confidence + 0.1)
                    evidence += "; good entry angle"
            elif outcome == "missed":
                if entry_angle < 30 or entry_angle > 60:
                    confidence = min(1.0, confidence + 0.1)
                    evidence += "; poor entry angle"

        return outcome, confidence, evidence

    def load_hoop_roi(self, config_path: str) -> bool:
        """Load normalized hoop region from YAML."""
        try:
            with open(config_path, 'r') as f:
                self.hoop_roi = yaml.safe_load(f)
                return True
        except FileNotFoundError:
            print(f"Warning: Hoop ROI config {config_path} not found")
            self.hoop_roi = None
            return False