"""Convert basketball videos + labels.csv into form feature arrays for the MLP.

Form-first: training target is ``class_id`` only. Optional ``made`` / ``has_hoop``
and ``--with-ball`` side metrics are written to meta / extra npz keys and never
replace the form label.

Run from the repo root::

    python -m ml.datasets.build_features_from_videos \\
        --videos ml/datasets/videos \\
        --labels ml/datasets/videos/labels.csv \\
        --output ml/datasets/data/train.npz

Optional ball/rim side stats (meta only)::

    ... --with-ball

See ``ml/docs/FORM_ML_AND_RULES.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python import vision

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ball.fusion import OutcomeFusion
from ball.trajectory import TrajectoryAnalyzer
from config.settings import DEFAULT_FPS
from phase_detection.phases import PHASE_ORDER
from pipeline import FrameResult, ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.timestamps import frame_timestamp_ms

ANGLE_JOINTS = (
    "elbow",
    "knee",
    "ankle_flexion",
    "hip",
    "shoulder",
    "index_align",
    "trunk",
)
KINEMATIC_FIELDS = (
    "wrist_velocity_y",
    "ankle_velocity_y",
    "hip_velocity_y",
    "nose_velocity_y",
    "index_velocity_y",
    "total_velocity",
    "elbow_angle",
    "knee_angle",
)


def feature_names() -> List[str]:
    names: List[str] = []
    for joint in ANGLE_JOINTS:
        names.append(f"angle_{joint}_mean")
        names.append(f"angle_{joint}_std")
    for field in KINEMATIC_FIELDS:
        names.append(f"kin_{field}_mean")
    for phase in PHASE_ORDER:
        names.append(f"phase_frac_{phase}")
    names.extend(["duration_sec", "n_frames_norm", "shooting_side_right"])
    return names


FEATURE_DIM = len(feature_names())


@dataclass
class LabelRow:
    video_path: str
    shot_index: Optional[int]  # None means all shots (*)
    class_id: int
    notes: str = ""
    made: Optional[int] = None  # 0 / 1 / None
    has_hoop: Optional[int] = None


@dataclass
class ExtractedShot:
    vector: np.ndarray
    rule_score: Optional[int]
    start_ms: int
    end_ms: int
    ball_meta: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build form ML feature arrays from basketball videos"
    )
    parser.add_argument(
        "--videos",
        default="ml/datasets/videos",
        help="Folder used to resolve relative video_path entries",
    )
    parser.add_argument(
        "--labels",
        default="ml/datasets/videos/labels.csv",
        help="CSV of video_path,shot_index,class_id[,made,has_hoop,notes]",
    )
    parser.add_argument(
        "--output",
        default="ml/datasets/data/train.npz",
        help="Output .npz path (keys: features, labels)",
    )
    parser.add_argument(
        "--auto-label-from-score",
        action="store_true",
        help="If a detected shot has no CSV label, map rule score -> class_id",
    )
    parser.add_argument(
        "--with-ball",
        action="store_true",
        help="Run ball/rim YOLO and store side metrics in meta (not form labels)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=8,
        help="Skip shots with fewer pose frames than this",
    )
    return parser.parse_args()


def _resolve_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidates = [
        (base / path).resolve(),
        (_REPO_ROOT / path).resolve(),
        (base / path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _parse_shot_index(raw: str) -> Optional[int]:
    text = (raw or "").strip().lower()
    if text in {"*", "-1", "all", ""}:
        return None
    return int(text)


def _parse_optional_01(raw: Optional[str]) -> Optional[int]:
    text = (raw or "").strip().lower()
    if text in {"", "na", "none", "unknown", "-"}:
        return None
    if text in {"1", "true", "yes", "made", "in"}:
        return 1
    if text in {"0", "false", "no", "miss", "missed", "out"}:
        return 0
    try:
        value = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Expected blank/0/1 for optional flag, got {raw!r}") from exc
    if value not in (0, 1):
        raise ValueError(f"Expected blank/0/1 for optional flag, got {raw!r}")
    return value


def load_labels(labels_path: Path) -> List[LabelRow]:
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels CSV not found: {labels_path}")

    rows: List[LabelRow] = []
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "shot_index", "class_id"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{labels_path} must have columns: video_path,shot_index,class_id "
                f"(optional: made,has_hoop,notes)"
            )
        for row in reader:
            video_path = (row.get("video_path") or "").strip()
            if not video_path or video_path.startswith("#"):
                continue
            class_id = int(row["class_id"])
            if class_id not in (0, 1):
                raise ValueError(
                    f"Binary form training requires class_id 0 or 1, got "
                    f"{class_id} on CSV line {reader.line_num}."
                )
            rows.append(
                LabelRow(
                    video_path=video_path,
                    shot_index=_parse_shot_index(row.get("shot_index", "*")),
                    class_id=class_id,
                    notes=(row.get("notes") or "").strip(),
                    made=_parse_optional_01(row.get("made")),
                    has_hoop=_parse_optional_01(row.get("has_hoop")),
                )
            )
    if not rows:
        raise ValueError(f"No label rows found in {labels_path}")
    return rows


def group_labels_by_video(rows: Sequence[LabelRow]) -> Dict[str, List[LabelRow]]:
    grouped: Dict[str, List[LabelRow]] = {}
    for row in rows:
        grouped.setdefault(row.video_path, []).append(row)
    return grouped


def score_to_class(score: Optional[int]) -> int:
    """Bootstrap binary labels from rule score (human review is preferred)."""
    return 0 if score is not None and score >= 70 else 1


def _angle_degrees(angles: dict, name: str) -> float:
    result = angles.get(name)
    if result is None or not getattr(result, "is_valid", False):
        return np.nan
    degrees = getattr(result, "degrees", None)
    if degrees is None:
        return np.nan
    return float(degrees)


def _safe_float(value: Optional[float]) -> float:
    if value is None:
        return np.nan
    return float(value)


def vectorize_shot(frames: Sequence[FrameResult]) -> np.ndarray:
    """Aggregate one completed shot into a fixed-length float32 vector."""
    if not frames:
        raise ValueError("Cannot vectorize empty shot")

    side = frames[0].shooting_side or "right"
    angle_series: Dict[str, List[float]] = {joint: [] for joint in ANGLE_JOINTS}
    kin_series: Dict[str, List[float]] = {field: [] for field in KINEMATIC_FIELDS}
    phase_counts = {phase: 0 for phase in PHASE_ORDER}

    for frame in frames:
        if not frame.has_pose:
            continue
        for joint in ANGLE_JOINTS:
            key = joint if joint == "trunk" else f"{side}_{joint}"
            angle_series[joint].append(_angle_degrees(frame.angles, key))
        feats = frame.features
        if feats is not None:
            for field in KINEMATIC_FIELDS:
                kin_series[field].append(_safe_float(getattr(feats, field, None)))
        if frame.phase in phase_counts:
            phase_counts[frame.phase] += 1

    values: List[float] = []
    for joint in ANGLE_JOINTS:
        arr = np.asarray(angle_series[joint], dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            values.extend([0.0, 0.0])
        else:
            values.append(float(np.nanmean(arr)))
            values.append(float(np.nanstd(arr)))
    for field in KINEMATIC_FIELDS:
        arr = np.asarray(kin_series[field], dtype=np.float64)
        if arr.size == 0 or np.all(np.isnan(arr)):
            values.append(0.0)
        else:
            values.append(float(np.nanmean(arr)))

    n_pose = max(1, sum(phase_counts.values()))
    for phase in PHASE_ORDER:
        values.append(phase_counts[phase] / n_pose)

    t0 = frames[0].timestamp_ms
    t1 = frames[-1].timestamp_ms
    duration_sec = max(0.0, (t1 - t0) / 1000.0)
    values.append(duration_sec)
    values.append(min(1.0, len(frames) / 120.0))
    values.append(1.0 if side == "right" else 0.0)

    vector = np.asarray(values, dtype=np.float32)
    if vector.shape[0] != FEATURE_DIM:
        raise RuntimeError(
            f"Feature dim mismatch: got {vector.shape[0]}, expected {FEATURE_DIM}"
        )
    np.nan_to_num(vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def lookup_label_row(
    video_labels: Sequence[LabelRow],
    shot_index: int,
) -> Optional[LabelRow]:
    for row in video_labels:
        if row.shot_index is None or row.shot_index == shot_index:
            return row
    return None


def lookup_class(
    video_labels: Sequence[LabelRow],
    shot_index: int,
    rule_score: Optional[int],
    auto_label: bool,
) -> Tuple[Optional[int], Optional[LabelRow]]:
    row = lookup_label_row(video_labels, shot_index)
    if row is not None:
        return row.class_id, row
    if auto_label:
        return score_to_class(rule_score), None
    return None, None


def _peak_ball_speed(pipeline: ShotAnalysisPipeline, start_ms: int, end_ms: int) -> float:
    window = pipeline.ball_buffer.get_window(start_ms, end_ms)
    peak = 0.0
    for snap in window:
        vx, vy = snap.velocity_xy
        peak = max(peak, math.hypot(float(vx), float(vy)))
    return float(peak)


def _release_speed_proxy(frames: Sequence[FrameResult]) -> float:
    """Wrist / total velocity near release — proxy, not lab force."""
    peak = 0.0
    for frame in frames:
        if frame.features is None:
            continue
        if frame.phase in {"release", "follow_through", "jump"}:
            peak = max(
                peak,
                abs(float(frame.features.wrist_velocity_y)),
                abs(float(frame.features.total_velocity)),
            )
    if peak > 0:
        return peak
    for frame in frames:
        if frame.features is None:
            continue
        peak = max(peak, abs(float(frame.features.wrist_velocity_y)))
    return float(peak)


def compute_ball_meta(
    pipeline: ShotAnalysisPipeline,
    frames: Sequence[FrameResult],
    start_ms: int,
    end_ms: int,
    with_ball: bool,
) -> Dict[str, Any]:
    if not with_ball:
        return {
            "with_ball": False,
            "made_pred": None,
            "made_confidence": None,
            "ball_speed_peak": None,
            "path_fit_r2": None,
            "entry_angle_deg": None,
            "release_speed_proxy": None,
            "force_proxy": None,
            "rim_seen": False,
        }

    fusion = OutcomeFusion()
    post_ms = int(fusion.post_shot_capture_ms)
    outcome = fusion.finalize_shot_outcome(
        pipeline.ball_buffer,
        start_ms,
        end_ms + post_ms,
    )
    analyzer = TrajectoryAnalyzer()
    trajectory = analyzer.analyze_shot_window(pipeline.ball_buffer)

    made_pred: Optional[int]
    if outcome is None or outcome.result == "unknown":
        made_pred = None
    elif outcome.result == "made":
        made_pred = 1
    else:
        made_pred = 0

    ball_speed = _peak_ball_speed(pipeline, start_ms, end_ms + post_ms)
    release_proxy = _release_speed_proxy(frames)
    rim_seen = any(f.rim is not None for f in frames)

    return {
        "with_ball": True,
        "made_pred": made_pred,
        "made_confidence": None if outcome is None else float(outcome.confidence),
        "ball_speed_peak": ball_speed,
        "path_fit_r2": None if trajectory is None else trajectory.r_squared,
        "entry_angle_deg": None if trajectory is None else trajectory.entry_angle_deg,
        "release_speed_proxy": release_proxy,
        # Honest name: kinematic proxy, not force-plate newtons.
        "force_proxy": release_proxy,
        "rim_seen": rim_seen,
        "outcome_evidence": [] if outcome is None else list(outcome.evidence),
    }


def extract_shots_from_video(
    video_path: Path,
    min_frames: int,
    with_ball: bool,
) -> List[ExtractedShot]:
    """Return form vectors (+ optional ball meta) for each completed shot."""
    detector = PoseDetector(running_mode=vision.RunningMode.VIDEO)
    pipeline = ShotAnalysisPipeline(enable_ball=with_ball)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    pipeline.set_fps(fps)

    current_frames: List[FrameResult] = []
    completed: List[Tuple[List[FrameResult], Optional[int]]] = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = frame_timestamp_ms(frame_index, fps)
        detection = detector.detect_video_frame(rgb, timestamp_ms)
        h, w, _ = frame.shape
        bgr = frame if with_ball else None
        result = pipeline.process_frame(
            detection, w, h, timestamp_ms, bgr_frame=bgr
        )

        if result.has_pose and (result.shot_in_progress or result.shot_summary):
            current_frames.append(result)

        if result.shot_summary is not None:
            score = result.shot_summary.score
            if len(current_frames) >= min_frames:
                completed.append((list(current_frames), score))
            current_frames = []

        frame_index += 1

    cap.release()

    trailing = pipeline.finalize_session()
    if trailing is not None and len(current_frames) >= min_frames:
        completed.append((list(current_frames), trailing.score))

    extracted: List[ExtractedShot] = []
    for frames, score in completed:
        start_ms = int(frames[0].timestamp_ms)
        end_ms = int(frames[-1].timestamp_ms)
        ball_meta = compute_ball_meta(
            pipeline, frames, start_ms, end_ms, with_ball=with_ball
        )
        extracted.append(
            ExtractedShot(
                vector=vectorize_shot(frames),
                rule_score=score,
                start_ms=start_ms,
                end_ms=end_ms,
                ball_meta=ball_meta,
            )
        )

    pipeline.reset()
    return extracted


def build_dataset(
    labels: Sequence[LabelRow],
    videos_root: Path,
    min_frames: int,
    auto_label: bool,
    with_ball: bool,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    features: List[np.ndarray] = []
    class_ids: List[int] = []
    meta: List[dict] = []

    for video_key, video_labels in group_labels_by_video(labels).items():
        video_path = _resolve_path(video_key, videos_root)
        if not video_path.is_file():
            print(f"SKIP missing video: {video_path}")
            continue

        print(f"Processing {video_path.name} (with_ball={with_ball}) ...")
        shots = extract_shots_from_video(
            video_path, min_frames=min_frames, with_ball=with_ball
        )
        print(f"  detected shots: {len(shots)}")

        for local_index, shot in enumerate(shots):
            class_id, label_row = lookup_class(
                video_labels,
                shot_index=local_index,
                rule_score=shot.rule_score,
                auto_label=auto_label,
            )
            if class_id is None:
                print(
                    f"  skip shot {local_index}: no label "
                    f"(score={shot.rule_score})"
                )
                continue
            features.append(shot.vector)
            class_ids.append(int(class_id))
            entry = {
                "video": str(video_path),
                "shot_index": local_index,
                "class_id": int(class_id),
                "rule_score": shot.rule_score,
                "start_ms": shot.start_ms,
                "end_ms": shot.end_ms,
                "made_label": None if label_row is None else label_row.made,
                "has_hoop_label": None if label_row is None else label_row.has_hoop,
                "notes": "" if label_row is None else label_row.notes,
            }
            entry.update(shot.ball_meta)
            meta.append(entry)
            print(
                f"  kept shot {local_index}: class={class_id} "
                f"rule_score={shot.rule_score} "
                f"made_label={entry['made_label']}"
            )

    if not features:
        raise RuntimeError(
            "No labeled shots exported. Check videos, labels.csv, "
            "or pass --auto-label-from-score."
        )

    feature_array = np.stack(features, axis=0).astype(np.float32)
    label_array = np.asarray(class_ids, dtype=np.int64)
    return feature_array, label_array, meta


def main() -> int:
    args = parse_args()

    videos_root = Path(args.videos)
    if not videos_root.is_absolute():
        videos_root = (_REPO_ROOT / videos_root).resolve()

    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = (_REPO_ROOT / labels_path).resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (_REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path)
    features, class_ids, meta = build_dataset(
        labels=labels,
        videos_root=videos_root,
        min_frames=int(args.min_frames),
        auto_label=bool(args.auto_label_from_score),
        with_ball=bool(args.with_ball),
    )

    save_kwargs: Dict[str, Any] = {
        "features": features,
        "labels": class_ids,
        "feature_names": np.asarray(feature_names()),
    }
    # Optional human made labels aligned with rows (NaN = unknown).
    made_labels = np.asarray(
        [
            np.nan if m.get("made_label") is None else float(m["made_label"])
            for m in meta
        ],
        dtype=np.float32,
    )
    save_kwargs["made_labels"] = made_labels

    np.savez_compressed(output_path, **save_kwargs)
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "feature_dim": int(features.shape[1]),
                "num_samples": int(features.shape[0]),
                "num_classes": int(class_ids.max()) + 1,
                "feature_names": feature_names(),
                "form_target": "class_id",
                "with_ball": bool(args.with_ball),
                "note": (
                    "Form MLP trains on features+labels only. "
                    "made_labels / ball fields are side channel."
                ),
                "shots": meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== export complete (form-first) ===")
    print(f"output:       {output_path}")
    print(f"meta:         {meta_path}")
    print(f"samples:      {features.shape[0]}")
    print(f"feature_dim:  {features.shape[1]}")
    print(
        f"label counts: "
        f"{ {int(k): int(v) for k, v in zip(*np.unique(class_ids, return_counts=True))} }"
    )
    print("Train with: python -m ml.training.train")
    print("Design: ml/docs/FORM_ML_AND_RULES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
