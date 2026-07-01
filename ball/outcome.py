"""Make / miss / unknown classification (Phase 6c)."""

from typing import Optional

from ball.models import BallTrajectory, ShotOutcome
from ball.timeseries import BallTimeSeriesBuffer


class OutcomeClassifier:
    """Rule-based shot outcome from ball time-series and hoop ROI."""

    def __init__(self):
        # TODO: load hoop ROI from config/hoop_roi.yaml
        # TODO: load outcome thresholds from config/ball.yaml
        pass

    def classify(
        self,
        ball_buffer: BallTimeSeriesBuffer,
        trajectory: Optional[BallTrajectory] = None,
    ) -> ShotOutcome:
        """Return made | missed | unknown with confidence and evidence."""
        # TODO: check ball crossing hoop ROI from above
        # TODO: distinguish made (inside rim) vs missed (outside / rim bounce)
        # TODO: return unknown when confidence too low or ROI not configured
        pass

    def load_hoop_roi(self, config_path: str) -> bool:
        """Load normalized hoop region from YAML."""
        # TODO: parse config/hoop_roi.yaml
        pass
