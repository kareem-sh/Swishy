"""
Central analysis pipeline: detect -> filter -> gate -> angles -> phases -> rules -> score.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from analysis.engine import BiomechanicsEngine
from analysis.models import AnalysisResult
from angles.calculator import AngleCalculator, AngleResult
from ball.detector import BallDetector
from ball.models import BallDetection, BallHolder, BallSnapshot, RimDetection, ShotOutcome, ShooterCourtPosition
from ball.nano_tracker import NanoBallTracker, NanoRimTracker
from ball.rim_tracker import RimMotionSmoother, is_plausible_rim_correction
from ball.shot_state_machine import BallShotStateMachine, BallStateUpdate
from ball.timeseries import BallTimeSeriesBuffer
from ball.tracker import BallTracker
from ball.trajectory_overlay import ObservedTrajectoryRecorder
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
from player.ball_holder import BallHolderTracker, best_candidate_for_ball, pose_candidates_from_detection_result
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
    observed_ball_path_segments: List[List[Tuple[float, float]]] = field(
        default_factory=list
    )
    fitted_observed_ball_path: List[Tuple[float, float]] = field(
        default_factory=list
    )
    ball_holder: Optional[BallHolder] = None
    shooter: Optional[BallHolder] = None
    selected_pose_index: int = 0
    pose_candidate_count: int = 0
    shooter_release: Optional[ShooterCourtPosition] = None
    court_calibration_valid: bool = False
    court_calibration_state: str = "UNINITIALIZED"
    court_calibration_confidence: Optional[float] = None
    court_reprojection_error_px: Optional[float] = None
    court_inlier_count: Optional[int] = None
    court_reference_hoop: Optional[str] = None
    court_last_failure: Optional[str] = None
    court_outline_px: List[Tuple[float, float]] = field(default_factory=list)
    court_origin_px: Optional[Tuple[float, float]] = None
    court_x_axis_px: Optional[Tuple[float, float]] = None
    court_y_axis_px: Optional[Tuple[float, float]] = None
    shooter_trace_court_xy: List[Tuple[float, float]] = field(default_factory=list)


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
        court_cfg = load_yaml("court.yaml")

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
        # The tracker opens shots BACKWARDS from a detected release, so it needs
        # the recent past, and it re-scores the captured frames against the
        # refined coaching phases, so it needs this engine rather than one of
        # its own (which could carry a different player profile).
        self._shot_tracker.attach_history(self._frame_buffer)
        self._shot_tracker.attach_analyzer(self._biomechanics)
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
        self._prev_image: Optional[dict] = None
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
        trajectory_display_cfg = display_cfg.get("trajectory_overlay", {})
        self._observed_trajectory = ObservedTrajectoryRecorder(
            max_points=int(trajectory_display_cfg.get("observed_max_points", 120)),
            maximum_jump_rim_radii=float(
                trajectory_display_cfg.get("maximum_jump_rim_radii", 5.0)
            ),
            fit_min_points=int(trajectory_display_cfg.get("fit_min_points", 5)),
            fit_samples=int(trajectory_display_cfg.get("fit_samples", 60)),
            fit_outlier_threshold_rim_radii=float(
                trajectory_display_cfg.get(
                    "fit_outlier_threshold_rim_radii", 0.75
                )
            ),
        )

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
        shooter_cfg = ball_cfg.get("shooter_tracking", {})
        self._ball_holder_tracker = BallHolderTracker(
            confirm_frames=int(shooter_cfg.get("confirm_frames", 4)),
            switch_confirm_frames=int(shooter_cfg.get("switch_confirm_frames", 3)),
            lose_frames=int(shooter_cfg.get("lose_frames", 8)),
            candidate_threshold=float(shooter_cfg.get("candidate_threshold", 0.45)),
            confirm_threshold=float(shooter_cfg.get("confirm_threshold", 0.60)),
            switch_margin=float(shooter_cfg.get("switch_margin", 0.12)),
        )
        self._court_service = None
        self._court_native_calibration = None
        self._court_last_failure: Optional[str] = None
        self._court_calibration_state = "UNINITIALIZED"
        self._court_calibration_attempts = 0
        self._court_trace: deque[Tuple[float, float]] = deque(maxlen=120)
        self._selected_pose_index = 0
        self._release_records: List[ShooterCourtPosition] = []
        self._shot_release_counter = 0
        video_cfg = court_cfg.get("video_calibration", {})
        self._court_enabled = bool(court_cfg.get("enabled", False))
        self._court_startup_max_attempts = int(video_cfg.get("startup_max_attempts", 10))
        self._court_startup_frame_window = int(video_cfg.get("startup_frame_window", 120))
        self._court_attempt_interval_frames = max(1, int(video_cfg.get("attempt_interval_frames", 3)))
        self._court_freeze_after_valid = bool(video_cfg.get("freeze_after_valid", True))
        self._court_detector = None
        self._court_calibrator = None
        if self._court_enabled:
            try:
                from court.calibration import CourtCalibrator
                from court.detector import CourtDetector

                self._court_detector = CourtDetector()
                self._court_calibrator = CourtCalibrator()
            except Exception as exc:
                self._court_enabled = False
                self._court_calibration_state = "INVALID"
                self._court_last_failure = f"court calibration unavailable: {exc}"
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

        rim_tracking_cfg = ball_cfg.get("rim_tracking", {})
        self._rim_tracking_enabled = bool(
            self._ball_enabled and rim_tracking_cfg.get("enabled", False)
        )
        self._rim_yolo_correction_interval = max(
            1, int(rim_tracking_cfg.get("yolo_correction_interval", 10))
        )
        self._rim_yolo_min_confidence = float(
            rim_tracking_cfg.get("yolo_reinitialize_confidence", 0.15)
        )
        self._rim_tracking_min_confidence = float(
            rim_tracking_cfg.get("minimum_tracking_confidence", 0.10)
        )
        self._rim_max_correction_distance_px = float(
            rim_tracking_cfg.get("yolo_max_correction_distance_px", 120.0)
        )
        self._rim_max_correction_distance_scale = float(
            rim_tracking_cfg.get("yolo_max_correction_distance_scale", 1.5)
        )
        self._rim_max_width_change_ratio = float(
            rim_tracking_cfg.get("maximum_width_change_ratio", 1.5)
        )
        rim_center_y_fraction = float(
            ball_cfg.get("detector", {}).get("rim_center_y_fraction", 0.5)
        )
        self._rim_smoother = RimMotionSmoother(
            position_alpha=float(rim_tracking_cfg.get("position_alpha", 0.75)),
            size_alpha=float(rim_tracking_cfg.get("size_alpha", 0.50)),
            center_y_fraction=rim_center_y_fraction,
        )
        self._nano_rim_tracker: Optional[NanoRimTracker] = None
        self._last_rim_yolo_frame = -self._rim_yolo_correction_interval

        if self._rim_tracking_enabled:
            backbone_path = PROJECT_ROOT / rim_tracking_cfg.get(
                "backbone_path",
                visual_tracking_cfg.get(
                    "backbone_path",
                    "models/nanotrack/nanotrack_backbone_sim.onnx",
                ),
            )
            neckhead_path = PROJECT_ROOT / rim_tracking_cfg.get(
                "neckhead_path",
                visual_tracking_cfg.get(
                    "neckhead_path",
                    "models/nanotrack/nanotrack_head_sim.onnx",
                ),
            )
            try:
                self._nano_rim_tracker = NanoRimTracker(
                    backbone_path=backbone_path,
                    neckhead_path=neckhead_path,
                    minimum_box_size_px=float(
                        rim_tracking_cfg.get("minimum_box_size_px", 16)
                    ),
                    maximum_center_jump_px=float(
                        rim_tracking_cfg.get("maximum_center_jump_px", 220)
                    ),
                    initial_box_scale=float(
                        rim_tracking_cfg.get("initial_box_scale", 1.0)
                    ),
                    minimum_size_ratio=float(
                        rim_tracking_cfg.get("minimum_frame_size_ratio", 0.65)
                    ),
                    maximum_size_ratio=float(
                        rim_tracking_cfg.get("maximum_frame_size_ratio", 1.55)
                    ),
                    minimum_aspect_ratio=float(
                        rim_tracking_cfg.get("minimum_aspect_ratio", 0.35)
                    ),
                    maximum_aspect_ratio=float(
                        rim_tracking_cfg.get("maximum_aspect_ratio", 8.0)
                    ),
                    center_y_fraction=rim_center_y_fraction,
                    device=str(rim_tracking_cfg.get("device", "auto")),
                    cuda_fp16=bool(rim_tracking_cfg.get("cuda_fp16", False)),
                )
            except Exception as exc:
                # Keep periodic YOLO rim updates active as the fallback.
                print(f"Warning: NanoTrack rim tracking unavailable ({exc})")

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

        # NanoTrack supplies cheap per-frame ball and rim measurements. YOLO
        # periodically corrects both trackers and reacquires either after loss.
        ball: Optional[BallDetection] = None
        rim = self._last_rim
        tracked_rim: Optional[RimDetection] = None
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

        if self._nano_rim_tracker is not None and self._nano_rim_tracker.active:
            tracked_rim = self._nano_rim_tracker.update(
                frame=bgr_frame,
                frame_index=self._frame_index,
                timestamp_ms=timestamp_ms,
            )
            if tracked_rim is not None:
                rim = self._rim_smoother.update(tracked_rim)
                self._last_rim = rim

        frames_since_yolo = self._frame_index - self._last_yolo_frame
        rim_frames_since_yolo = self._frame_index - self._last_rim_yolo_frame
        rim_correction_due = (
            self._rim_tracking_enabled
            and rim_frames_since_yolo >= self._rim_yolo_correction_interval
        )
        run_yolo = (
            not self._visual_tracking_enabled
            or ball is None
            or frames_since_yolo >= self._yolo_correction_interval
            or rim_correction_due
        )

        if run_yolo:
            court = self._ball_detector.detect_court(
                bgr_frame,
                self._frame_index,
                timestamp_ms,
            )
            self._last_yolo_frame = self._frame_index
            self._last_rim_yolo_frame = self._frame_index

            if (
                court.rim is not None
                and court.rim.confidence >= self._rim_yolo_min_confidence
            ):
                rim_reference = tracked_rim or self._last_rim
                weak_or_reacquiring = (
                    tracked_rim is None
                    or tracked_rim.confidence < self._rim_tracking_min_confidence
                )
                accept_rim = (
                    rim_reference is None
                    or is_plausible_rim_correction(
                        court.rim,
                        rim_reference,
                        maximum_center_distance_px=(
                            float("inf")
                            if weak_or_reacquiring
                            else self._rim_max_correction_distance_px
                        ),
                        maximum_center_distance_scale=(
                            self._rim_max_correction_distance_scale
                        ),
                        maximum_width_change_ratio=(
                            self._rim_max_width_change_ratio
                        ),
                    )
                )
                if accept_rim:
                    reacquired = tracked_rim is None
                    if self._nano_rim_tracker is not None:
                        self._nano_rim_tracker.initialize(
                            frame=bgr_frame,
                            detection=court.rim,
                        )
                    rim = self._rim_smoother.update(
                        court.rim,
                        snap=(reacquired and self._nano_rim_tracker is not None),
                    )
                    self._last_rim = rim

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
        self._maybe_update_court_calibration(bgr_frame)
        ball, rim, ball_snapshot = self._process_ball(bgr_frame, timestamp_ms)

        hold_ball_xy = None
        if ball is not None:
            hold_ball_xy = ball.center_xy
        elif ball_snapshot is not None:
            hold_ball_xy = ball_snapshot.center_xy
        hold_ball_confidence = (
            ball.confidence
            if ball is not None
            else (ball_snapshot.confidence if ball_snapshot is not None else 0.0)
        )

        pose_candidates = pose_candidates_from_detection_result(
            detection_result,
            width,
            height,
        )
        pose_candidate_count = len(pose_candidates)
        chosen_candidate = best_candidate_for_ball(
            pose_candidates,
            hold_ball_xy,
            hold_ball_confidence,
        )
        if chosen_candidate is not None:
            self._selected_pose_index = chosen_candidate.player_id
        elif self._ball_holder_tracker.current is not None:
            self._selected_pose_index = self._ball_holder_tracker.current.player_id
        else:
            self._selected_pose_index = 0

        raw = extract_all_landmarks(
            detection_result,
            width,
            height,
            pose_index=self._selected_pose_index,
        )

        if raw is None:
            ball_state_update = self._ball_shot_fsm.update(
                ball_detection=ball,
                ball_snapshot=ball_snapshot,
                rim_detection=rim,
                ankle_y= None,
                wrist_xy=None,
                pose_phase=None,
                timestamp_ms=timestamp_ms,
            )
            self._update_observed_trajectory(ball_snapshot, ball_state_update)
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
            ball_holder = self._ball_holder_tracker.update(
                hold_ball_xy,
                pose_candidates,
                ball_confidence=hold_ball_confidence,
                court_service=self._court_service,
                released=bool(ball_state_update.released_this_frame),
            )
            self._update_shooter_trace(ball_holder)
            shooter_release = self._maybe_capture_release_position(
                ball_state_update,
                ball_holder,
            )
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
                ball_holder=ball_holder,
                shooter=ball_holder,
                selected_pose_index=self._selected_pose_index,
                pose_candidate_count=pose_candidate_count,
                shooter_release=shooter_release,
                **self._court_frame_fields(),
                shooter_trace_court_xy=list(self._court_trace),
                **self._ball_state_frame_fields(ball_state_update),
            )
        left_ankle_y_px = raw["image"]["left_ankle"]["y"]
        right_ankle_y_px = raw["image"]["right_ankle"]["y"]
        ankle_y_px = (left_ankle_y_px + right_ankle_y_px) / 2.0
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
            prev_image_landmarks=self._prev_image,
        )
        self._prev_image = raw["image"]

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
        # print("ankle" , ankle_y_px , " , " , "basket_y" , ball_snapshot.y)
        ball_state_update = self._ball_shot_fsm.update(
            ball_detection=ball,
            ball_snapshot=ball_snapshot,
            rim_detection=rim,
            wrist_xy=wrist_xy,
            ankle_y=ankle_y_px,
            pose_phase=phase,
            timestamp_ms=timestamp_ms,
            player_height_px=(
                features.body_pixel_height * height
                if features.body_pixel_height > 0.0
                else None
            ),
        )
        self._update_observed_trajectory(ball_snapshot, ball_state_update)
        ball_holder = self._ball_holder_tracker.update(
            hold_ball_xy,
            pose_candidates,
            ball_confidence=hold_ball_confidence,
            court_service=self._court_service,
            released=bool(ball_state_update.released_this_frame),
        )
        self._update_shooter_trace(ball_holder)
        shooter_release = self._maybe_capture_release_position(
            ball_state_update,
            ball_holder,
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
            ball_holder=ball_holder,
            shooter=ball_holder,
            selected_pose_index=self._selected_pose_index,
            pose_candidate_count=pose_candidate_count,
            shooter_release=shooter_release,
            **self._court_frame_fields(),
            shooter_trace_court_xy=list(self._court_trace),
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

    def _update_observed_trajectory(
        self,
        ball_snapshot: Optional[BallSnapshot],
        update: BallStateUpdate,
    ) -> None:
        self._observed_trajectory.update(
            ball_snapshot,
            released_this_frame=update.released_this_frame,
            shot_finished=update.outcome is not None,
            rim_center_xy=update.rim_center_xy,
            rim_radius=update.rim_inner_radius,
        )

    def _ball_state_frame_fields(self, update: BallStateUpdate) -> dict:
        return {
            "ball_state": update.state.value,
            "ball_tracking_status": update.tracking_status.value,
            "shot_outcome": update.outcome,
            "stabilized_rim_center_xy": update.rim_center_xy,
            "stabilized_rim_inner_radius": update.rim_inner_radius,
            "rim_crossing_xy": update.crossing_xy,
            "observed_ball_path_segments": (
                self._observed_trajectory.screen_segments(
                    update.rim_center_xy,
                    update.rim_inner_radius,
                )
            ),
            "fitted_observed_ball_path": (
                self._observed_trajectory.fitted_screen_curve(
                    update.rim_center_xy,
                    update.rim_inner_radius,
                )
            ),
        }

    def _court_frame_fields(self) -> dict:
        calibration = self._court_native_calibration
        outline: list[Tuple[float, float]] = []
        origin_px = None
        x_axis_px = None
        y_axis_px = None
        reference_hoop = None

        if self._court_service is not None and calibration is not None:
            try:
                outline = [
                    self._court_service.court_to_image((0.0, 0.0)),
                    self._court_service.court_to_image((0.0, 28.0)),
                    self._court_service.court_to_image((7.5, 28.0)),
                    self._court_service.court_to_image((7.5, 0.0)),
                    self._court_service.court_to_image((-7.5, 0.0)),
                    self._court_service.court_to_image((-7.5, 28.0)),
                ]
                origin_px = self._court_service.court_to_image((0.0, 0.0))
                x_axis_px = self._court_service.court_to_image((3.0, 0.0))
                y_axis_px = self._court_service.court_to_image((0.0, 3.0))
                reference_hoop = self._court_service.calibration.court_frame.reference_hoop
            except (RuntimeError, ValueError):
                outline = []

        return {
            "court_calibration_valid": self._court_service is not None,
            "court_calibration_state": self._court_calibration_state,
            "court_calibration_confidence": (
                None
                if self._court_service is None
                else self._court_service.calibration.confidence
            ),
            "court_reprojection_error_px": (
                None if calibration is None else calibration.reprojection_error_px
            ),
            "court_inlier_count": None if calibration is None else calibration.inlier_count,
            "court_reference_hoop": reference_hoop,
            "court_last_failure": self._court_last_failure,
            "court_outline_px": [(float(x), float(y)) for x, y in outline],
            "court_origin_px": None if origin_px is None else (float(origin_px[0]), float(origin_px[1])),
            "court_x_axis_px": None if x_axis_px is None else (float(x_axis_px[0]), float(x_axis_px[1])),
            "court_y_axis_px": None if y_axis_px is None else (float(y_axis_px[0]), float(y_axis_px[1])),
        }

    def _maybe_update_court_calibration(self, bgr_frame: Optional[np.ndarray]) -> None:
        if not self._court_enabled or bgr_frame is None:
            return
        if self._court_service is not None and self._court_freeze_after_valid:
            self._court_calibration_state = "VALID"
            return
        if self._court_calibration_attempts >= self._court_startup_max_attempts:
            if self._court_service is None:
                self._court_calibration_state = "INVALID"
            return
        if self._frame_index > self._court_startup_frame_window:
            if self._court_service is None:
                self._court_calibration_state = "INVALID"
                if not self._court_last_failure:
                    self._court_last_failure = "startup frame window exhausted"
            return
        if self._frame_index % self._court_attempt_interval_frames != 0:
            return
        if self._court_detector is None or self._court_calibrator is None:
            self._court_calibration_state = "INVALID"
            if not self._court_last_failure:
                self._court_last_failure = "court detector unavailable"
            return

        self._court_calibration_state = "CALIBRATING"
        self._court_calibration_attempts += 1
        try:
            from court.swishy_calibration import adapt_from_config

            detection = self._court_detector.detect(bgr_frame)
            native = self._court_calibrator.calibrate(bgr_frame, detection)
            swishy = adapt_from_config(native)
        except Exception as exc:
            self._court_last_failure = str(exc)
            if self._court_calibration_attempts >= self._court_startup_max_attempts:
                self._court_calibration_state = "INVALID"
            else:
                self._court_calibration_state = "UNINITIALIZED"
            return

        if swishy.calibration.status == "VALID":
            self._court_service = swishy
            self._court_native_calibration = native
            self._court_calibration_state = "VALID"
            self._court_last_failure = None
            return

        self._court_last_failure = swishy.calibration.message
        self._court_native_calibration = native
        if self._court_calibration_attempts >= self._court_startup_max_attempts:
            self._court_calibration_state = "INVALID"
        else:
            self._court_calibration_state = "UNINITIALIZED"

    def _maybe_capture_release_position(
        self,
        update: BallStateUpdate,
        ball_holder: Optional[BallHolder],
    ) -> Optional[ShooterCourtPosition]:
        if not update.released_this_frame or ball_holder is None:
            return None

        self._shot_release_counter += 1
        calibration = self._court_native_calibration
        court_x_m = court_y_m = distance_m = None
        if ball_holder.court_position is not None:
            court_x_m, court_y_m, _ = ball_holder.court_position
            distance_m = ball_holder.distance_to_hoop_m

        image_x = image_y = None
        if ball_holder.image_position is not None:
            image_x, image_y = ball_holder.image_position

        status = "VALID"
        if self._court_calibration_state != "VALID":
            status = "COURT_UNCALIBRATED"
        elif court_x_m is None or court_y_m is None:
            status = "COURT_POSITION_UNAVAILABLE"
        elif ball_holder.tracking_status != "CONFIDENT":
            status = "SHOOTER_TENTATIVE"

        release = ShooterCourtPosition(
            shooter_id=ball_holder.player_id,
            release_frame=self._frame_index,
            release_timestamp_ms=update.outcome.release_timestamp_ms
            if update.outcome is not None and update.outcome.release_timestamp_ms is not None
            else (self._prev_timestamp_ms or 0),
            confidence=ball_holder.confidence,
            image_x_px=image_x,
            image_y_px=image_y,
            court_x_m=court_x_m,
            court_y_m=court_y_m,
            court_z_m=0.0,
            distance_to_hoop_m=distance_m,
            status=status,
            court_calibration_valid=self._court_calibration_state == "VALID",
            court_reprojection_error_px=(
                None if calibration is None else calibration.reprojection_error_px
            ),
            court_inliers=None if calibration is None else calibration.inlier_count,
        )
        self._release_records.append(release)
        return release

    def _update_shooter_trace(self, ball_holder: Optional[BallHolder]) -> None:
        if ball_holder is None or ball_holder.court_position is None:
            return
        self._court_trace.append((ball_holder.court_position[0], ball_holder.court_position[1]))

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
        self._prev_image = None
        self._prev_timestamp_ms = None
        self._ankle_baseline_y = 0.0
        self._ankle_image_baseline = 0.0
        self._ball_tracker.reset()
        self._ball_buffer.clear()
        self._ball_shot_fsm.reset()
        self._ball_holder_tracker.reset()
        self._observed_trajectory.reset()
        self._last_rim = None
        self._last_yolo_frame = -1
        self._court_trace.clear()
        self._selected_pose_index = 0
        self._release_records.clear()
        self._shot_release_counter = 0
        self._court_service = None
        self._court_native_calibration = None
        self._court_last_failure = None
        self._court_calibration_attempts = 0
        self._court_calibration_state = "UNINITIALIZED" if self._court_enabled else "INVALID"
        self._last_rim_yolo_frame = -self._rim_yolo_correction_interval
        self._rim_smoother.reset()
        if self._nano_ball_tracker is not None:
            self._nano_ball_tracker.reset()
        if self._nano_rim_tracker is not None:
            self._nano_rim_tracker.reset()
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
    def court_service(self):
        return self._court_service

    @property
    def court_native_calibration(self):
        return self._court_native_calibration

    @property
    def court_calibration_state(self) -> str:
        return self._court_calibration_state

    @property
    def release_records(self) -> List[ShooterCourtPosition]:
        return list(self._release_records)

    @property
    def shooter_switches(self) -> int:
        return self._ball_holder_tracker.shooter_switches

    @property
    def ball_device(self):
        """Configured ball/rim inference device, or None when disabled."""
        if self._ball_detector is None:
            return None
        return self._ball_detector.device
