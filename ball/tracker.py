"""Multi-frame ball tracking (Phase 6a)."""

from typing import List, Optional
import numpy as np

from ball.models import BallDetection, BallSnapshot


class BallTracker:
    """Maintain ball identity and smooth position across frames."""

    def __init__(self):
        self.track_id = 0
        self.track_history = []
        self.current_position = None
        self.last_detection_frame = -1
        self.max_gap_frames = 2
        
        # Kalman filter parameters (simplified)
        self.kalman = None
        self._init_kalman()

    def _init_kalman(self):
        """Initialize Kalman filter for tracking."""
        # Simple moving average filter instead of full Kalman
        self.position_history = []
        self.velocity_history = []
        self.max_history = 5

    def update(
        self,
        detection: Optional[BallDetection],
        frame_index: int,
        timestamp_ms: int,
    ) -> Optional[BallSnapshot]:
        """Update tracker state; return smoothed snapshot or None."""
        
        if detection is not None:
            # Update with new detection
            self.current_position = (detection.x, detection.y)
            self.last_detection_frame = frame_index
            self.position_history.append((detection.x, detection.y, timestamp_ms))
            
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
            
            # Smooth position
            if len(self.position_history) >= 3:
                smooth_x = np.mean([p[0] for p in self.position_history[-3:]])
                smooth_y = np.mean([p[1] for p in self.position_history[-3:]])
            else:
                smooth_x, smooth_y = detection.x, detection.y
            
            # Create snapshot
            snapshot = BallSnapshot(
                x=int(smooth_x),
                y=int(smooth_y),
                radius=detection.radius,
                confidence=detection.confidence,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                track_id=self.track_id
            )
            
            self.track_history.append(snapshot)
            return snapshot
            
        else:
            # No detection - interpolate if gap is small
            if (frame_index - self.last_detection_frame) <= self.max_gap_frames:
                # Interpolate from last known position
                if len(self.track_history) >= 2:
                    last = self.track_history[-1]
                    prev = self.track_history[-2] if len(self.track_history) >= 2 else last
                    
                    # Linear interpolation
                    dt = 1.0  # Assuming 1 frame gap
                    vx = (last.x - prev.x) / (last.frame_index - prev.frame_index) if last.frame_index != prev.frame_index else 0
                    vy = (last.y - prev.y) / (last.frame_index - prev.frame_index) if last.frame_index != prev.frame_index else 0
                    
                    interp_x = int(last.x + vx * dt)
                    interp_y = int(last.y + vy * dt)
                    
                    snapshot = BallSnapshot(
                        x=interp_x,
                        y=interp_y,
                        radius=last.radius,
                        confidence=last.confidence * 0.5,  # Lower confidence for interpolated
                        frame_index=frame_index,
                        timestamp_ms=timestamp_ms,
                        track_id=self.track_id,
                        is_interpolated=True
                    )
                    
                    self.track_history.append(snapshot)
                    return snapshot
            
            return None

    def reset(self):
        """Clear track state (new shot or new session)."""
        self.track_id += 1
        self.track_history = []
        self.current_position = None
        self.last_detection_frame = -1
        self.position_history = []
        self.velocity_history = []

    def get_track(self) -> List[BallSnapshot]:
        """Return full track history for current shot."""
        return self.track_history.copy()

    def get_current_position(self) -> Optional[tuple]:
        """Get current ball position."""
        return self.current_position