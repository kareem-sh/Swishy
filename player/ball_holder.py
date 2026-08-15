"""Ball-holder association and shooter tracking over pose candidates."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Tuple

from ball.models import BallHolder, BallHolderStatus, PlayerPoseCandidate, ShooterSelectionState
from pose.landmarks import BASKETBALL_LANDMARKS


Coord2 = Tuple[float, float]


def _distance(a: Optional[Coord2], b: Optional[Coord2]) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: Optional[Coord2], b: Optional[Coord2]) -> Optional[Coord2]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _landmark_xy(landmarks: dict[str, Any], key: str) -> Optional[Coord2]:
    landmark = landmarks.get(key)
    if landmark is None:
        return None
    visibility = float(landmark.get("visibility", 1.0))
    presence = float(landmark.get("presence", 1.0))
    if visibility < 0.25 or presence < 0.25:
        return None
    return float(landmark["x"]), float(landmark["y"])


def pose_candidates_from_detection_result(detection_result, width: int, height: int) -> list[PlayerPoseCandidate]:
    """Convert MediaPipe pose detections into scored shooter candidates."""
    if detection_result is None or not getattr(detection_result, "pose_landmarks", None):
        return []

    candidates: list[PlayerPoseCandidate] = []
    for player_id, pose_landmarks in enumerate(detection_result.pose_landmarks):
        pose_image = {}
        for name, index in BASKETBALL_LANDMARKS.items():
            landmark = pose_landmarks[index]
            pose_image[name] = {
                "x": float(landmark.x * width),
                "y": float(landmark.y * height),
                "visibility": float(landmark.visibility),
                "presence": float(landmark.presence),
            }
        candidates.append(_candidate_from_pose_image(player_id, pose_image))
    return candidates


def best_candidate_for_ball(
    candidates: Sequence[PlayerPoseCandidate],
    ball_xy: Optional[Coord2],
    ball_confidence: float = 0.0,
) -> Optional[PlayerPoseCandidate]:
    """Return the strongest ball-associated pose candidate for this frame."""
    if not candidates:
        return None
    if ball_xy is None:
        return max(candidates, key=lambda candidate: candidate.confidence)
    return max(
        candidates,
        key=lambda candidate: _score_candidate(candidate, ball_xy, ball_confidence),
    )


def _candidate_from_pose_image(player_id: int, pose_image: dict[str, dict[str, float]]) -> PlayerPoseCandidate:
    left_wrist = _landmark_xy(pose_image, "left_wrist")
    right_wrist = _landmark_xy(pose_image, "right_wrist")
    left_ankle = _landmark_xy(pose_image, "left_ankle")
    right_ankle = _landmark_xy(pose_image, "right_ankle")
    left_heel = _landmark_xy(pose_image, "left_heel")
    right_heel = _landmark_xy(pose_image, "right_heel")
    left_foot_index = _landmark_xy(pose_image, "left_foot_index")
    right_foot_index = _landmark_xy(pose_image, "right_foot_index")
    left_hip = _landmark_xy(pose_image, "left_hip")
    right_hip = _landmark_xy(pose_image, "right_hip")
    left_shoulder = _landmark_xy(pose_image, "left_shoulder")
    right_shoulder = _landmark_xy(pose_image, "right_shoulder")

    feet_midpoint = _midpoint(left_foot_index, right_foot_index)
    if feet_midpoint is None:
        feet_midpoint = _midpoint(left_heel, right_heel)
    if feet_midpoint is None:
        feet_midpoint = _midpoint(left_ankle, right_ankle)
    if feet_midpoint is None:
        feet_midpoint = _midpoint(left_hip, right_hip)

    body_center = _midpoint(
        _midpoint(left_shoulder, right_shoulder),
        _midpoint(left_hip, right_hip),
    )
    if body_center is None:
        body_center = feet_midpoint

    all_points = [
        point
        for point in (
            left_wrist,
            right_wrist,
            left_ankle,
            right_ankle,
            left_heel,
            right_heel,
            left_foot_index,
            right_foot_index,
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        )
        if point is not None
    ]
    bbox = None
    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        bbox = (min(xs), min(ys), max(xs), max(ys))

    visible_count = sum(1 for point in (left_wrist, right_wrist, left_ankle, right_ankle, left_hip, right_hip) if point is not None)
    pose_confidence = visible_count / 6.0

    return PlayerPoseCandidate(
        player_id=player_id,
        image_center_xy=body_center,
        left_wrist_xy=left_wrist,
        right_wrist_xy=right_wrist,
        feet_midpoint_xy=feet_midpoint,
        left_ankle_xy=left_ankle,
        right_ankle_xy=right_ankle,
        left_heel_xy=left_heel,
        right_heel_xy=right_heel,
        body_center_xy=body_center,
        bbox_xyxy=bbox,
        confidence=max(0.0, min(1.0, pose_confidence)),
        tracking_status=BallHolderStatus.UNKNOWN.value,
    )


class BallHolderTracker:
    """Track a stable shooter identity with hysteresis."""

    def __init__(
        self,
        *,
        confirm_frames: int = 4,
        switch_confirm_frames: int = 3,
        lose_frames: int = 8,
        candidate_threshold: float = 0.45,
        confirm_threshold: float = 0.60,
        switch_margin: float = 0.12,
    ):
        self.confirm_frames = max(1, int(confirm_frames))
        self.switch_confirm_frames = max(1, int(switch_confirm_frames))
        self.lose_frames = max(1, int(lose_frames))
        self.candidate_threshold = float(candidate_threshold)
        self.confirm_threshold = float(confirm_threshold)
        self.switch_margin = float(switch_margin)

        self.current: Optional[BallHolder] = None
        self._current_state = ShooterSelectionState.NO_SHOOTER
        self._candidate_id: Optional[int] = None
        self._candidate_streak = 0
        self._lost_frames = 0
        self._switch_count = 0

    @property
    def shooter_switches(self) -> int:
        return self._switch_count

    def reset(self) -> None:
        self.current = None
        self._current_state = ShooterSelectionState.NO_SHOOTER
        self._candidate_id = None
        self._candidate_streak = 0
        self._lost_frames = 0
        self._switch_count = 0

    def update(
        self,
        ball_xy: Optional[Coord2],
        pose_candidates: Sequence[PlayerPoseCandidate],
        *,
        ball_confidence: float = 0.0,
        court_service: Any = None,
        released: bool = False,
    ) -> Optional[BallHolder]:
        if ball_xy is None or not pose_candidates:
            return self._handle_missing_ball_or_pose(released=released)

        self._lost_frames = 0
        scored = sorted(
            (
                (_score_candidate(candidate, ball_xy, ball_confidence), candidate)
                for candidate in pose_candidates
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_candidate = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        evidence_gap = best_score - runner_up

        if best_score < self.candidate_threshold:
            return self._handle_missing_ball_or_pose(released=released)

        selected_holder = _convert_candidate_to_holder(
            best_candidate,
            ball_xy,
            base_score=best_score,
            court_service=court_service,
        )

        if self.current is None:
            if self._candidate_id == best_candidate.player_id:
                self._candidate_streak += 1
            else:
                self._candidate_id = best_candidate.player_id
                self._candidate_streak = 1
            self.current = selected_holder
            if self._candidate_streak >= self.confirm_frames and best_score >= self.confirm_threshold:
                self._current_state = ShooterSelectionState.CONFIRMED_SHOOTER
                selected_holder.tracking_status = BallHolderStatus.CONFIDENT.value
            else:
                self._current_state = ShooterSelectionState.CANDIDATE
                selected_holder.tracking_status = BallHolderStatus.TENTATIVE.value
            selected_holder.shooter_state = self._state_for_frame(released)
            return selected_holder

        if best_candidate.player_id == self.current.player_id:
            self._candidate_id = best_candidate.player_id
            self._candidate_streak = min(self._candidate_streak + 1, 100000)
            selected_holder.confidence = min(1.0, selected_holder.confidence + 0.12)
            if selected_holder.confidence >= self.confirm_threshold or self._candidate_streak >= self.confirm_frames:
                self._current_state = ShooterSelectionState.CONFIRMED_SHOOTER
                selected_holder.tracking_status = BallHolderStatus.CONFIDENT.value
            else:
                self._current_state = ShooterSelectionState.CANDIDATE
                selected_holder.tracking_status = BallHolderStatus.TENTATIVE.value
            selected_holder.shooter_state = self._state_for_frame(released)
            self.current = selected_holder
            return self.current

        if evidence_gap >= self.switch_margin and best_score >= self.confirm_threshold:
            if self._candidate_id == best_candidate.player_id:
                self._candidate_streak += 1
            else:
                self._candidate_id = best_candidate.player_id
                self._candidate_streak = 1

            if self._candidate_streak >= self.switch_confirm_frames:
                self._switch_count += 1
                self.current = selected_holder
                self._current_state = ShooterSelectionState.CONFIRMED_SHOOTER
                self.current.tracking_status = BallHolderStatus.CONFIDENT.value
                self.current.shooter_state = self._state_for_frame(released)
                return self.current

        self.current.tracking_status = BallHolderStatus.TENTATIVE.value
        self.current.shooter_state = self._state_for_frame(released)
        return self.current

    def _handle_missing_ball_or_pose(self, *, released: bool) -> Optional[BallHolder]:
        self._lost_frames += 1
        if self.current is None:
            self._current_state = ShooterSelectionState.NO_SHOOTER
            return None
        if self._lost_frames >= self.lose_frames:
            self.current.tracking_status = BallHolderStatus.LOST.value
            self._current_state = ShooterSelectionState.NO_SHOOTER
            self.current.shooter_state = ShooterSelectionState.NO_SHOOTER.value
            return self.current
        self.current.tracking_status = BallHolderStatus.TENTATIVE.value
        self.current.shooter_state = self._state_for_frame(released)
        return self.current

    def _state_for_frame(self, released: bool) -> str:
        if released:
            return ShooterSelectionState.RELEASED.value
        if self._current_state == ShooterSelectionState.CONFIRMED_SHOOTER:
            return ShooterSelectionState.CONFIRMED_SHOOTER.value
        if self._current_state == ShooterSelectionState.CANDIDATE:
            return ShooterSelectionState.CANDIDATE.value
        if self._current_state == ShooterSelectionState.NO_SHOOTER:
            return ShooterSelectionState.NO_SHOOTER.value
        return ShooterSelectionState.UNKNOWN.value


def _score_candidate(candidate: PlayerPoseCandidate, ball_xy: Coord2, ball_confidence: float) -> float:
    score = 0.0
    score += 0.30 * max(0.0, min(1.0, candidate.confidence))

    left_dist = _distance(ball_xy, candidate.left_wrist_xy)
    right_dist = _distance(ball_xy, candidate.right_wrist_xy)
    hand_dist = min(left_dist, right_dist)
    if math.isfinite(hand_dist):
        score += 0.45 * max(0.0, 1.0 - hand_dist / 200.0)

    body_dist = _distance(ball_xy, candidate.body_center_xy)
    if math.isfinite(body_dist):
        score += 0.20 * max(0.0, 1.0 - body_dist / 300.0)

    if candidate.feet_midpoint_xy is not None:
        score += 0.05

    score += 0.10 * max(0.0, min(1.0, ball_confidence))
    return max(0.0, min(1.0, score))


def _choose_ground_point(candidate: PlayerPoseCandidate) -> tuple[Optional[Coord2], Optional[Coord2], Optional[Coord2]]:
    foot_mid = candidate.feet_midpoint_xy
    if foot_mid is not None:
        return foot_mid, candidate.left_heel_xy or candidate.left_ankle_xy, candidate.right_heel_xy or candidate.right_ankle_xy
    heel_mid = _midpoint(candidate.left_heel_xy, candidate.right_heel_xy)
    if heel_mid is not None:
        return heel_mid, candidate.left_heel_xy, candidate.right_heel_xy
    ankle_mid = _midpoint(candidate.left_ankle_xy, candidate.right_ankle_xy)
    return ankle_mid, candidate.left_ankle_xy, candidate.right_ankle_xy


def _convert_candidate_to_holder(
    candidate: PlayerPoseCandidate,
    ball_xy: Coord2,
    *,
    base_score: float,
    court_service: Any = None,
) -> BallHolder:
    left_wrist = candidate.left_wrist_xy
    right_wrist = candidate.right_wrist_xy
    if left_wrist is not None and right_wrist is not None:
        nearest_wrist = left_wrist if _distance(ball_xy, left_wrist) <= _distance(ball_xy, right_wrist) else right_wrist
    else:
        nearest_wrist = left_wrist or right_wrist

    ground_xy, left_foot_xy, right_foot_xy = _choose_ground_point(candidate)

    court_point = None
    if court_service is not None and ground_xy is not None:
        try:
            court_point = court_service.image_to_court(ground_xy)
        except (RuntimeError, ValueError):
            court_point = None

    holder = BallHolder(
        player_id=candidate.player_id,
        confidence=max(0.0, min(1.0, base_score)),
        image_position=ground_xy or candidate.image_center_xy,
        left_wrist_xy=left_wrist,
        right_wrist_xy=right_wrist,
        feet_midpoint_xy=ground_xy,
        left_foot_xy=left_foot_xy,
        right_foot_xy=right_foot_xy,
        nearest_wrist_xy=nearest_wrist,
        bbox_xyxy=candidate.bbox_xyxy,
        court_position=None if court_point is None else (court_point.x_m, court_point.y_m, court_point.z_m),
        tracking_status=BallHolderStatus.TENTATIVE.value,
        shooter_state=ShooterSelectionState.CANDIDATE.value,
        distance_to_hoop_m=None,
        signed_x_offset_m=None,
        signed_y_distance_m=None,
    )

    if holder.court_position is not None:
        x_m, y_m, _ = holder.court_position
        holder.distance_to_hoop_m = math.hypot(x_m, y_m)
        holder.signed_x_offset_m = float(x_m)
        holder.signed_y_distance_m = float(y_m)

    return holder
