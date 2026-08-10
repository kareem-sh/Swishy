"""
Central analysis pipeline: detect -> filter -> gate -> angles -> phases -> rules -> score.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from analysis.engine import BiomechanicsEngine
from analysis.models import AnalysisResult
from angles.calculator import AngleCalculator, AngleResult
from ball.detector import BallDetector
from ball.models import BallDetection, BallSnapshot, RimDetection, ShotOutcome
from ball.nano_tracker import NanoBallTracker
from ball.shot_state_machine import BallShotStateMachine, BallStateUpdate
from ball.timeseries import BallTimeSeriesBuffer
from ball.tracker import BallTracker
from config.settings import (
    DEFAULT_FPS,
    FILTER_BETA,
    FILTER_D_CUTOFF,
    FILTER_MIN_CUTOFF,
    FRAME_BUFFER_SIZE,
    PRESENCE_THRESHOLD,
    PROJECT_ROOT,
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
    update_ankle_image_baseline,
)
from phase_detection.phases import PHASE_LABELS
from player.profile import PlayerProfile, load_player_profile
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
    body_capture_active: bool = False
    last_shot_score: Optional[int] = None
    shot_summary: Optional[ShotSummary] = None
    display_summary: Optional[ShotSummary] = None
    show_shot_summary: bool = False
    hud_display: Optional["HudDisplay"] = None
    capture_warning: Optional[str] = None
    # Phase 6 — ball / rim (custom basketball YOLO)
    ball: Optional[BallDetection] = None
    rim: Optional[RimDetection] = None
    ball_snapshot: Optional[BallSnapshot] = None
    ball_state: str = "waiting"
    ball_tracking_status: str = "lost"
    shot_outcome: Optional[ShotOutcome] = None
    stabilized_rim_center_xy: Optional[Tuple[float, float]] = None
    stabilized_rim_inner_radius: Optional[float] = None
    rim_crossing_xy: Optional[Tuple[float, float]] = None


class ShotAnalysisPipeline:
    """
    End-to-end per-frame processing for basketball shooting analysis.

    Stages:
        0. Ball + rim YOLO (optional, parallel to pose)
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

    def __init__(
        self,
        shooting_hand: Optional[str] = None,
        enable_ball: Optional[bool] = None,
        player: Optional[PlayerProfile] = None,
    ):
        # Player context is session-level and optional. A profile without a
        # height is a first-class state: height-independent analysis still runs.
        self._player = player if player is not None else load_player_profile()

        filter_cfg = load_yaml("filter_config.yaml")
        phase_cfg = load_yaml("phases.yaml")
        scoring_cfg = load_yaml("scoring.yaml")
        display_cfg = load_yaml("display.yaml")
        ball_cfg = load_yaml("ball.yaml")

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
        self._biomechanics = BiomechanicsEngine(self._player)
        self._shot_tracker = ShotTracker()
        self._frame_buffer = FrameBuffer(max_frames=FRAME_BUFFER_SIZE)
        # Explicit argument wins, then the player profile, then settings.py.
        self._shooting_hand = (
            shooting_hand
            if shooting_hand is not None
            else (
                self._player.shooting_hand
                if self._player.shooting_hand != "auto"
                else SHOOTING_HAND
            )
        )
        self._resolved_side = "right"
        self._frame_index = 0
        self._fps = DEFAULT_FPS
        self._prev_timestamp_ms: Optional[int] = None
        self._prev_world: Optional[dict] = None
        self._ankle_baseline_y = 0.0
        self._ankle_image_baseline = 0.0
        self._still_threshold = float(
            phase_cfg.get("thresholds", {}).get("ready_max_velocity", 0.15)
        )
        self._summary_display_frames = int(
            display_cfg.get("summary_display_frames", scoring_cfg.get("summary_display_frames", 90))
        )
        self._hud_display = HudDisplaySmoother()
        self._show_ball_overlay = bool(display_cfg.get("show_ball_overlay", True))

        # Phase 6 — custom basketball YOLO (ball + hoop)
        # enable_ball=False skips YOLO load (used by ML feature export).
        if enable_ball is None:
            self._ball_enabled = bool(ball_cfg.get("enabled", False))
        else:
            self._ball_enabled = bool(enable_ball)
        self._ball_detector: Optional[BallDetector] = None
        tracking_cfg = ball_cfg.get("tracking", {})
        self._ball_tracker = BallTracker(
            max_gap_frames=int(tracking_cfg.get("max_missing_frames", 4)),
            position_alpha=float(tracking_cfg.get("position_alpha", 0.80)),
            velocity_alpha=float(tracking_cfg.get("velocity_alpha", 0.50)),
        )
        self._ball_buffer = BallTimeSeriesBuffer()
        self._ball_shot_fsm = BallShotStateMachine("ball.yaml")
        self._last_rim: Optional[RimDetection] = None
        self._last_yolo_frame = -1

        visual_tracking_cfg = ball_cfg.get("visual_tracking", {})
        self._visual_tracking_enabled = bool(
            self._ball_enabled and visual_tracking_cfg.get("enabled", False)
        )
        self._yolo_correction_interval = max(
            1, int(visual_tracking_cfg.get("yolo_correction_interval", 10))
        )
        self._yolo_reinitialize_confidence = float(
            visual_tracking_cfg.get("yolo_reinitialize_confidence", 0.05)
        )
        self._yolo_override_confidence = float(
            visual_tracking_cfg.get("yolo_override_confidence", 0.15)
        )
        self._yolo_max_correction_distance_px = float(
            visual_tracking_cfg.get("yolo_max_correction_distance_px", 180.0)
        )
        self._nano_ball_tracker: Optional[NanoBallTracker] = None

        if self._visual_tracking_enabled:
            backbone_path = PROJECT_ROOT / visual_tracking_cfg.get(
                "backbone_path",
                "models/nanotrack/nanotrack_backbone_sim.onnx",
            )
            neckhead_path = PROJECT_ROOT / visual_tracking_cfg.get(
                "neckhead_path",
                "models/nanotrack/nanotrack_head_sim.onnx",
            )
            try:
                self._nano_ball_tracker = NanoBallTracker(
                    backbone_path=backbone_path,
                    neckhead_path=neckhead_path,
                    minimum_box_size_px=float(
                        visual_tracking_cfg.get("minimum_box_size_px", 12)
                    ),
                    maximum_center_jump_px=float(
                        visual_tracking_cfg.get("maximum_center_jump_px", 180)
                    ),
                    initial_box_scale=float(
                        visual_tracking_cfg.get("initial_box_scale", 1.35)
                    ),
                    device=str(visual_tracking_cfg.get("device", "auto")),
                    cuda_fp16=bool(visual_tracking_cfg.get("cuda_fp16", False)),
                )
            except Exception as exc:
                print(f"Warning: NanoTrack disabled ({exc})")
                self._visual_tracking_enabled = False

        if self._ball_enabled:
            try:
                detector = BallDetector("ball.yaml")
                self._ball_detector = detector if detector.ready else None
            except Exception as exc:
                print(f"Warning: ball/rim detector disabled ({exc})")
                self._ball_detector = None

        state_cfg = ball_cfg.get("ball_state_machine", {})
        self._shot_tracker.configure_ball_outcome(
            required=bool(
                self._ball_detector is not None and self._ball_shot_fsm.enabled
            ),
            body_grace_ms=int(state_cfg.get("body_grace_ms", 500)),
        )

    def set_fps(self, fps: float):
        if fps > 0:
            self._fps = fps

    def _process_ball(
        self,
        bgr_frame: Optional[np.ndarray],
        timestamp_ms: int,
    ) -> Tuple[
        Optional[BallDetection],
        Optional[RimDetection],
        Optional[BallSnapshot],
    ]:
        if self._ball_detector is None or bgr_frame is None:
            return None, self._last_rim, None

        # NanoTrack supplies the cheap per-frame measurement. YOLO acquires the
        # ball, corrects drift periodically, and reacquires it after a loss.
        ball: Optional[BallDetection] = None
        rim = self._last_rim
        if (
            self._visual_tracking_enabled
            and self._nano_ball_tracker is not None
            and self._nano_ball_tracker.active
        ):
            ball = self._nano_ball_tracker.update(
                frame=bgr_frame,
                frame_index=self._frame_index,
                timestamp_ms=timestamp_ms,
            )

        frames_since_yolo = self._frame_index - self._last_yolo_frame
        run_yolo = (
            not self._visual_tracking_enabled
            or ball is None
            or frames_since_yolo >= self._yolo_correction_interval
        )

        if run_yolo:
            court = self._ball_detector.detect_court(
                bgr_frame,
                self._frame_index,
                timestamp_ms,
            )
            self._last_yolo_frame = self._frame_index

            if court.rim is not None:
                rim = court.rim
                self._last_rim = court.rim

            if (
                court.ball is not None
                and court.ball.confidence
                >= self._yolo_reinitialize_confidence
            ):
                accept_yolo = ball is None
                if ball is not None:
                    center_distance = float(
                        np.linalg.norm(
                            np.asarray(court.ball.center_xy)
                            - np.asarray(ball.center_xy)
                        )
                    )
                    accept_yolo = (
                        court.ball.confidence >= self._yolo_override_confidence
                        or center_distance <= self._yolo_max_correction_distance_px
                    )

                if accept_yolo:
                    ball = court.ball
                    if (
                        self._visual_tracking_enabled
                        and self._nano_ball_tracker is not None
                    ):
                        self._nano_ball_tracker.initialize(
                            frame=bgr_frame,
                            detection=court.ball,
                        )

        snapshot = self._ball_tracker.update(
            ball,
            self._frame_index,
            timestamp_ms,
        )

        if snapshot is not None:
            self._ball_buffer.push(snapshot)

        return ball, rim, snapshot

    def process_frame(
        self,
        detection_result,
        width: int,
        height: int,
        timestamp_ms: int,
        bgr_frame: Optional[np.ndarray] = None,
    ) -> FrameResult:
        self._frame_index += 1
        timestamp_s = timestamp_ms / 1000.0
        ball, rim, ball_snapshot = self._process_ball(bgr_frame, timestamp_ms)

        raw = extract_all_landmarks(detection_result, width, height)
        if raw is None:
            ball_state_update = self._ball_shot_fsm.update(
                ball_detection=ball,
                ball_snapshot=ball_snapshot,
                rim_detection=rim,
                wrist_xy=None,
                pose_phase=None,
                timestamp_ms=timestamp_ms,
            )
            completed_shot = self._shot_tracker.update_ball_outcome(
                ball_state_update.outcome,
                timestamp_ms,
            )
            if completed_shot is not None:
                self._shot_tracker.begin_summary_display(
                    self._summary_display_frames
                )
            display_summary = None
            if self._shot_tracker.show_shot_summary:
                display_summary = completed_shot or self._shot_tracker.last_summary
            phase = self._phase_detector.phase
            return FrameResult(
                timestamp_ms=timestamp_ms,
                has_pose=False,
                shooting_side=self._resolved_side,
                phase=phase,
                phase_label=PHASE_LABELS.get(phase, phase),
                shot_in_progress=self._shot_tracker.shot_in_progress,
                body_capture_active=self._shot_tracker.capture_in_progress,
                last_shot_score=self._shot_tracker.last_score,
                shot_summary=completed_shot,
                display_summary=display_summary,
                show_shot_summary=self._shot_tracker.show_shot_summary,
                capture_warning=self._shot_tracker.capture_warning,
                ball=ball,
                rim=rim,
                ball_snapshot=ball_snapshot,
                **self._ball_state_frame_fields(ball_state_update),
            )

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
            image_landmarks=raw["image"],
            ankle_image_baseline=self._ankle_image_baseline,
        )

        self._ankle_baseline_y = update_ankle_baseline(
            self._ankle_baseline_y,
            ankle_y,
            features.total_velocity,
            self._still_threshold,
        )
        self._ankle_image_baseline = update_ankle_image_baseline(
            self._ankle_image_baseline,
            raw["image"],
            features.total_velocity,
            self._still_threshold,
        )
        features.ankle_baseline_y = self._ankle_baseline_y

        phase = self._phase_detector.update(features, timestamp_ms=timestamp_ms)
        phase_label = PHASE_LABELS.get(phase, phase)
        analysis = self._biomechanics.evaluate(phase, angles, features, shooting_side)

        wrist_xy = self._shooting_wrist_xy(raw["image"], shooting_side)
        ball_state_update = self._ball_shot_fsm.update(
            ball_detection=ball,
            ball_snapshot=ball_snapshot,
            rim_detection=rim,
            wrist_xy=wrist_xy,
            pose_phase=phase,
            timestamp_ms=timestamp_ms,
        )

        snapshot = FrameSnapshot(
            timestamp_ms=timestamp_ms,
            angles=angles,
            shooting_side=shooting_side,
            phase=phase,
            features=features,
            analysis=analysis,
        )
        self._frame_buffer.push(snapshot)

        completed_shot = self._shot_tracker.update(
            phase,
            snapshot,
            ball_outcome=ball_state_update.outcome,
        )
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
            body_capture_active=self._shot_tracker.capture_in_progress,
            last_shot_score=self._shot_tracker.last_score,
            shot_summary=completed_shot,
            display_summary=display_summary,
            show_shot_summary=self._shot_tracker.show_shot_summary,
            hud_display=hud_display,
            capture_warning=self._shot_tracker.capture_warning,
            ball=ball,
            rim=rim,
            ball_snapshot=ball_snapshot,
            **self._ball_state_frame_fields(ball_state_update),
        )

    @staticmethod
    def _shooting_wrist_xy(
        image_landmarks: dict,
        shooting_side: str,
    ) -> Optional[Tuple[float, float]]:
        wrist = image_landmarks.get(f"{shooting_side}_wrist")
        if wrist is None:
            return None
        if float(wrist.get("visibility", 0.0)) < 0.5:
            return None
        if float(wrist.get("presence", 0.0)) < 0.5:
            return None
        return float(wrist["x"]), float(wrist["y"])

    @staticmethod
    def _ball_state_frame_fields(update: BallStateUpdate) -> dict:
        return {
            "ball_state": update.state.value,
            "ball_tracking_status": update.tracking_status.value,
            "shot_outcome": update.outcome,
            "stabilized_rim_center_xy": update.rim_center_xy,
            "stabilized_rim_inner_radius": update.rim_inner_radius,
            "rim_crossing_xy": update.crossing_xy,
        }

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
        self._ankle_image_baseline = 0.0
        self._ball_tracker.reset()
        self._ball_buffer.clear()
        self._ball_shot_fsm.reset()
        self._last_rim = None
        self._last_yolo_frame = -1
        if self._nano_ball_tracker is not None:
            self._nano_ball_tracker.reset()
        if self._ball_detector is not None:
            self._ball_detector.reset()

    @property
    def player(self) -> PlayerProfile:
        """Session-level player context (height, handedness)."""
        return self._player

    @property
    def frame_buffer(self) -> FrameBuffer:
        return self._frame_buffer

    @property
    def shot_tracker(self) -> ShotTracker:
        return self._shot_tracker

    @property
    def ball_buffer(self) -> BallTimeSeriesBuffer:
        return self._ball_buffer

    @property
    def ball_enabled(self) -> bool:
        return self._ball_detector is not None

    @property
    def show_ball_overlay(self) -> bool:
        return self._show_ball_overlay and self.ball_enabled

    @property
    def ball_device(self):
        """Configured ball/rim inference device, or None when disabled."""
        if self._ball_detector is None:
            return None
        return self._ball_detector.device
