from collections import deque
from dataclasses import dataclass

from typing import Deque, Dict, List, Optional



from angles.calculator import AngleResult

from analysis.models import AnalysisResult

from phase_detection.features import KinematicFeatures





@dataclass

class FrameSnapshot:

    timestamp_ms: int

    angles: Dict[str, AngleResult]

    shooting_side: str

    phase: str = "ready_stance"

    features: Optional[KinematicFeatures] = None

    analysis: Optional[AnalysisResult] = None





class FrameBuffer:

    """Ring buffer for temporal analysis (phase detection, shot scoring)."""



    def __init__(self, max_frames: int = 300):

        self.max_frames = max(1, int(max_frames))

        self._frames: Deque[FrameSnapshot] = deque(maxlen=self.max_frames)



    def push(self, snapshot: FrameSnapshot):

        self._frames.append(snapshot)



    @property

    def frames(self) -> List[FrameSnapshot]:

        return list(self._frames)



    @property

    def latest(self) -> Optional[FrameSnapshot]:

        return self._frames[-1] if self._frames else None



    @property

    def previous(self) -> Optional[FrameSnapshot]:

        if len(self._frames) < 2:

            return None

        return self._frames[-2]



    def clear(self):

        self._frames.clear()


