"""
Central analysis pipeline: detect -> filter -> gate -> angles -> phases -> rules -> score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from analysis.engine import BiomechanicsEngine
from analysis.models import AnalysisResult
from angles.calculator import AngleCalculator, AngleResult
from config.settings import (
    DEFAULT_FPS,
    FILTER_BETA,
    FILTER_D_CUTOFF,
    FILTER_MIN_CUTOFF,
    FRAME_BUFFER_SIZE,
    PRESENCE_THRESHOLD,
    SHOOTING_HAND,
    VISIBILITY_HOLD_FRAMES,
    VISIBILITY_REQUIRE_PRESENCE,
    VISIBILITY_THRESHOLD,
)
from feedback.models import ShotSummary
from feedback.shot_tracker import ShotTracker
from filters.one_euro import LandmarkFilterBank
from phase_detection.detector import ShotPhaseDetector
from phase_detection.features import (
    KinematicFeatures,
    _avg_y,
    extract_features,
    update_ankle_baseline,
)
from phase_detection.phases import PHASE_LABELS
from pose.landmarks import extract_all_landmarks
from pose.visibility import VisibilityGate
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameBuffer, FrameSnapshot
from visualization.hud_display import HudDisplay, HudDisplaySmoother


@dataclass
class FrameResult:
    """Output of processing a single frame through the pipeline."""

    image_landmarks: Optional[dict] = None
    world_landmarks: Optional[dict] = None
    angles: Dict[str, AngleResult] = field(default_factory=dict)
    features: Optional[KinematicFeatures] = None
    analysis: Optional[AnalysisResult] = None
    shooting_side: str = "right"
    phase: str = "ready_stance"
    phase_label: str = "Ready Stance"
    timestamp_ms: int = 0
    has_pose: bool = False
    shot_in_progress: bool = False
    last_shot_score: Optional[int] = None
    shot_summary: Optional[ShotSummary] = None
    display_summary: Optional[ShotSummary] = None
    show_shot_summary: bool = False
    hud_display: Optional["HudDisplay"] = None
    capture_warning: Optional[str] = None


class ShotAnalysisPipeline:
    """
    End-to-end per-frame processing for basketball shooting analysis.

    Stages:
        1. Extract image + world landmarks
        2. One Euro filter on world positions
        3. Visibility gating
        4. 3D joint angle computation
        5. Kinematic feature extraction
        6. Phase detection (FSM)
        7. Biomechanical rule evaluation
        8. Shot tracking + scoring (Phase 5)
        9. Temporal buffer storage
    """

    def __init__(self, shooting_hand: str = SHOOTING_HAND):
        filter_cfg = load_yaml("filter_config.yaml")
        phase_cfg = load_yaml("phases.yaml")
        scoring_cfg = load_yaml("scoring.yaml")
        display_cfg = load_yaml("display.yaml")

        self._filter_bank = LandmarkFilterBank(
            min_cutoff=filter_cfg.get("min_cutoff", FILTER_MIN_CUTOFF),
            beta=filter_cfg.get("beta", FILTER_BETA),
            d_cutoff=filter_cfg.get("d_cutoff", FILTER_D_CUTOFF),
        )
        self._visibility = VisibilityGate(
            visibility_threshold=VISIBILITY_THRESHOLD,
            presence_threshold=PRESENCE_THRESHOLD,
            hold_frames=VISIBILITY_HOLD_FRAMES,
            require_presence=VISIBILITY_REQUIRE_PRESENCE,
        )
        self._angle_calculator = AngleCalculator(self._visibility)
        self._phase_detector = ShotPhaseDetector()
        self._biomechanics = BiomechanicsEngine()
        self._shot_tracker = ShotTracker()
        self._frame_buffer = FrameBuffer(max_frames=FRAME_BUFFER_SIZE)
        self._shooting_hand = shooting_hand
        self._resolved_side = "right"
        self._frame_index = 0
        self._fps = DEFAULT_FPS
        self._prev_timestamp_ms: Optional[int] = None
        self._prev_world: Optional[dict] = None
        self._ankle_baseline_y = 0.0
        self._still_threshold = float(
            phase_cfg.get("thresholds", {}).get("ready_max_velocity", 0.15)
        )
        self._summary_display_frames = int(
            display_cfg.get("summary_display_frames", scoring_cfg.get("summary_display_frames", 90))
        )
        self._hud_display = HudDisplaySmoother()

    def set_fps(self, fps: float):
        if fps > 0:
            self._fps = fps

    def process_frame(self, detection_result, width: int, height: int, timestamp_ms: int) -> FrameResult:
        self._frame_index += 1
        timestamp_s = timestamp_ms / 1000.0

        raw = extract_all_landmarks(detection_result, width, height)
        if raw is None:
            return FrameResult(timestamp_ms=timestamp_ms, has_pose=False)

        world = self._filter_bank.filter_landmarks(raw["world"], timestamp_s)
        world = self._visibility.apply(world)

        shooting_side = self._resolve_shooting_side(world)
        angles = self._angle_calculator.compute_all(world, shooting_side)

        dt_s = self._compute_dt(timestamp_ms)
        prev_snapshot = self._frame_buffer.latest

        ankle_y = 0.0
        ankle_avg = _avg_y(world, ("left_ankle", "right_ankle"))
        if ankle_avg is not None:
            ankle_y = ankle_avg

        features = extract_features(
            world_landmarks=world,
            angles=angles,
            shooting_side=shooting_side,
            prev_world=self._prev_world,
            prev_angles=prev_snapshot.angles if prev_snapshot else None,
            dt_s=dt_s,
            ankle_baseline_y=self._ankle_baseline_y,
        )

        self._ankle_baseline_y = update_ankle_baseline(
            self._ankle_baseline_y,
            ankle_y,
            features.total_velocity,
            self._still_threshold,
        )
        features.ankle_baseline_y = self._ankle_baseline_y

        phase = self._phase_detector.update(features)
        phase_label = PHASE_LABELS.get(phase, phase)
        analysis = self._biomechanics.evaluate(phase, angles, features, shooting_side)

        snapshot = FrameSnapshot(
            timestamp_ms=timestamp_ms,
            angles=angles,
            shooting_side=shooting_side,
            phase=phase,
            features=features,
            analysis=analysis,
        )
        self._frame_buffer.push(snapshot)

        completed_shot = self._shot_tracker.update(phase, snapshot)
        if completed_shot is not None:
            self._shot_tracker.begin_summary_display(self._summary_display_frames)

        display_summary = None
        if self._shot_tracker.show_shot_summary:
            display_summary = completed_shot or self._shot_tracker.last_summary

        self._prev_world = world
        self._prev_timestamp_ms = timestamp_ms

        hud_display = self._hud_display.update(
            FrameResult(
                has_pose=True,
                phase=phase,
                phase_label=phase_label,
                shooting_side=shooting_side,
                angles=angles,
                analysis=analysis,
                shot_in_progress=self._shot_tracker.shot_in_progress,
                last_shot_score=self._shot_tracker.last_score,
                display_summary=display_summary,
            )
        )

        return FrameResult(
            image_landmarks=raw["image"],
            world_landmarks=world,
            angles=angles,
            features=features,
            analysis=analysis,
            shooting_side=shooting_side,
            phase=phase,
            phase_label=phase_label,
            timestamp_ms=timestamp_ms,
            has_pose=True,
            shot_in_progress=self._shot_tracker.shot_in_progress,
            last_shot_score=self._shot_tracker.last_score,
            shot_summary=completed_shot,
            display_summary=display_summary,
            show_shot_summary=self._shot_tracker.show_shot_summary,
            hud_display=hud_display,
            capture_warning=self._shot_tracker.capture_warning,
        )

    def _compute_dt(self, timestamp_ms: int) -> float:
        if self._prev_timestamp_ms is not None and timestamp_ms > self._prev_timestamp_ms:
            return (timestamp_ms - self._prev_timestamp_ms) / 1000.0
        return 1.0 / self._fps

    def _resolve_shooting_side(self, world_landmarks: dict) -> str:
        if self._shooting_hand in ("left", "right"):
            self._resolved_side = self._shooting_hand
            return self._resolved_side

        left_wrist = world_landmarks.get("left_wrist")
        right_wrist = world_landmarks.get("right_wrist")

        if left_wrist and right_wrist and left_wrist.get("is_reliable") and right_wrist.get("is_reliable"):
            if left_wrist["position"][1] > right_wrist["position"][1]:
                self._resolved_side = "left"
            else:
                self._resolved_side = "right"

        return self._resolved_side

    def finalize_session(self) -> Optional[ShotSummary]:
        """Close any shot still in progress (e.g. video ended mid-rep)."""
        return self._shot_tracker.finalize_in_progress()

    def reset(self):
        self._filter_bank.reset()
        self._visibility.reset()
        self._phase_detector.reset()
        self._shot_tracker.reset()
        self._hud_display.reset()
        self._frame_buffer.clear()
        self._frame_index = 0
        self._prev_world = None
        self._prev_timestamp_ms = None
        self._ankle_baseline_y = 0.0

    @property
    def frame_buffer(self) -> FrameBuffer:
        return self._frame_buffer

    @property
    def shot_tracker(self) -> ShotTracker:
        return self._shot_tracker
