"""Record shots and key frames across a full session (video, live, or image)."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

import numpy as np

from feedback.console import print_shot_summary
from feedback.frame_capture import KeyFrameCapture
from feedback.models import ShotSummary
from feedback.report_builder import build_detailed_shot_report, build_session_report
from feedback.report_models import DetailedShotReport, SessionReport
from feedback.report_writer import write_session_report
from pipeline import FrameResult
from utils.config_loader import load_yaml

if TYPE_CHECKING:
    from pipeline import ShotAnalysisPipeline


class SessionRecorder:
    """
    Collects per-shot data and key frames for detailed reports.

    Works for video, live webcam sessions, and single images.
    """

    def __init__(self, source_type: str, source_name: str, fps: float = 30.0):
        self.source_type = source_type
        self.source_name = source_name
        self.fps = fps
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
        self.total_frames = 0

        cfg = load_yaml("report_config.yaml")
        self._max_key_frames = int(cfg.get("max_key_frames_per_shot", 12))
        self._store_all_shot_frames = bool(cfg.get("store_frame_images_during_shot", True))
        self._auto_save_report = bool(cfg.get("auto_save_report", True))

        self._capture: Optional[KeyFrameCapture] = None
        self._all_frame_images: Dict[int, np.ndarray] = {}
        self._phase_moments: List[dict] = []
        self._prev_phase: Optional[str] = None
        self._shot_start_ms = 0
        self._shot_frames: List[FrameResult] = []
        self._shots: List[DetailedShotReport] = []

    def on_frame(
        self,
        frame_index: int,
        annotated_bgr: np.ndarray,
        frame_result: FrameResult,
    ):
        self.total_frames = frame_index + 1

        if frame_result.shot_in_progress and self._capture is None:
            self._start_shot(frame_result)

        if self._capture and frame_result.shot_in_progress:
            if frame_result.phase != self._prev_phase:
                self._phase_moments.append({
                    "phase": frame_result.phase,
                    "phase_label": frame_result.phase_label,
                    "timestamp_ms": frame_result.timestamp_ms,
                    "frame_index": frame_index,
                })
                self._prev_phase = frame_result.phase

            self._shot_frames.append(frame_result)
            if self._store_all_shot_frames:
                self._all_frame_images[frame_index] = annotated_bgr.copy()
            self._capture.consider(frame_index, annotated_bgr, frame_result)

        if frame_result.shot_summary:
            self._finish_shot(frame_result.shot_summary)

    def on_single_image(self, frame_result: FrameResult, annotated_bgr: np.ndarray):
        """Build a minimal single-frame report for image mode."""
        from feedback.generator import generate_coaching_tips
        from feedback.scorer import score_shot
        from utils.frame_buffer import FrameSnapshot

        snapshot = FrameSnapshot(
            timestamp_ms=0,
            angles=frame_result.angles,
            shooting_side=frame_result.shooting_side,
            phase=frame_result.phase,
            features=frame_result.features,
            analysis=frame_result.analysis,
        )
        summary = score_shot([snapshot], shot_number=1)
        summary.coaching_tips = generate_coaching_tips(summary)

        capture = KeyFrameCapture(max_frames=1)
        fr_copy = FrameResult(
            has_pose=True,
            phase=frame_result.phase,
            phase_label=frame_result.phase_label,
            timestamp_ms=0,
            shooting_side=frame_result.shooting_side,
            angles=frame_result.angles,
            analysis=frame_result.analysis,
            shot_in_progress=True,
        )
        capture.consider(0, annotated_bgr, fr_copy)
        key_frame_pairs = capture.finalize()

        detailed = build_detailed_shot_report(
            summary=summary,
            phase_moments=[{
                "phase": frame_result.phase,
                "phase_label": frame_result.phase_label,
                "timestamp_ms": 0,
                "frame_index": 0,
            }],
            key_frame_pairs=key_frame_pairs,
            start_ms=0,
            end_ms=0,
        )
        self._shots.append(detailed)
        self._all_frame_images.update(detailed.frame_images)
        self.total_frames = 1

    def finalize(
        self,
        output_dir: Optional[Path] = None,
        pipeline: Optional["ShotAnalysisPipeline"] = None,
    ) -> SessionReport:
        if pipeline is not None:
            orphan = pipeline.finalize_session()
            if orphan is not None:
                print_shot_summary(orphan)
                self._finish_shot(orphan)

        report = build_session_report(
            session_id=self.session_id,
            source_type=self.source_type,
            source_name=self.source_name,
            fps=self.fps,
            total_frames=self.total_frames,
            shots=self._shots,
        )

        save_dir = output_dir
        if save_dir is None and self._auto_save_report:
            from config.settings import REPORT_OUTPUT_DIR
            save_dir = Path(REPORT_OUTPUT_DIR)

        if save_dir is not None:
            frame_images = self._collect_frame_images()
            path = write_session_report(report, frame_images, save_dir)
            report.output_path = str(path)

        return report

    def _collect_frame_images(self) -> Dict[int, np.ndarray]:
        images = dict(self._all_frame_images)
        for shot in self._shots:
            images.update(shot.frame_images)
        return images

    def _start_shot(self, frame_result: FrameResult):
        self._capture = KeyFrameCapture(max_frames=self._max_key_frames)
        self._phase_moments = []
        self._shot_frames = []
        self._shot_start_ms = frame_result.timestamp_ms
        self._prev_phase = frame_result.phase
        self._phase_moments.append({
            "phase": frame_result.phase,
            "phase_label": frame_result.phase_label,
            "timestamp_ms": frame_result.timestamp_ms,
            "frame_index": self.total_frames - 1,
        })

    def _finish_shot(self, summary: ShotSummary):
        if self._capture is None:
            return

        key_frame_pairs = self._capture.finalize()
        end_ms = self._shot_frames[-1].timestamp_ms if self._shot_frames else self._shot_start_ms

        detailed = build_detailed_shot_report(
            summary=summary,
            phase_moments=self._phase_moments,
            key_frame_pairs=key_frame_pairs,
            start_ms=self._shot_start_ms,
            end_ms=end_ms,
        )
        self._shots.append(detailed)
        self._all_frame_images.update(detailed.frame_images)

        self._capture = None
        self._phase_moments = []
        self._shot_frames = []

    @property
    def shots(self) -> List[DetailedShotReport]:
        return self._shots
