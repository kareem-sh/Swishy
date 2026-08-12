"""Compare ball/rim inference settings without claiming ground-truth accuracy.

This measures raw detection coverage, confidence, continuity, and speed. True
precision/recall still requires labeled bounding boxes.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from ball.detector import BallDetector


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    imgsz: int
    ball_conf: float
    rim_conf: float


@dataclass
class DetectionStats:
    frames: int = 0
    ball_hits: int = 0
    rim_hits: int = 0
    ball_confidence: float = 0.0
    rim_confidence: float = 0.0
    ball_max_gap: int = 0
    rim_max_gap: int = 0
    ball_mean_jump: float = 0.0
    fps: float = 0.0


CONFIGS = (
    BenchmarkConfig("baseline", 512, 0.15, 0.10),
    BenchmarkConfig("640-balanced", 640, 0.12, 0.08),
    BenchmarkConfig("640-strict", 640, 0.15, 0.10),
    BenchmarkConfig("704-balanced", 704, 0.12, 0.08),
)


def run_config(
    detector: BallDetector,
    source: Path,
    config: BenchmarkConfig,
    max_frames: int,
) -> DetectionStats:
    detector.imgsz = config.imgsz
    detector.min_confidence = config.ball_conf
    detector.min_confidence_rim = config.rim_conf
    detector.frame_stride = 1
    detector.fallback_imgsz = 0
    detector.sticky_rim = False
    detector.reset()

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    stats = DetectionStats()
    ball_gap = rim_gap = 0
    ball_jumps = []
    previous_ball: Optional[tuple[float, float]] = None
    started = time.perf_counter()

    while stats.frames < max_frames:
        ok, frame = capture.read()
        if not ok:
            break

        detections = detector.detect_court(
            frame,
            stats.frames,
            int(stats.frames * 1000 / 30),
        )
        stats.frames += 1

        if detections.ball is not None:
            stats.ball_hits += 1
            stats.ball_confidence += detections.ball.confidence
            ball_gap = 0
            if previous_ball is not None:
                jump = math.dist(previous_ball, detections.ball.center_xy)
                ball_jumps.append(jump)
            previous_ball = detections.ball.center_xy
        else:
            ball_gap += 1
            stats.ball_max_gap = max(stats.ball_max_gap, ball_gap)

        if detections.rim is not None:
            stats.rim_hits += 1
            stats.rim_confidence += detections.rim.confidence
            rim_gap = 0
        else:
            rim_gap += 1
            stats.rim_max_gap = max(stats.rim_max_gap, rim_gap)

    capture.release()
    elapsed = time.perf_counter() - started
    stats.fps = stats.frames / max(elapsed, 1e-6)
    stats.ball_mean_jump = (
        sum(ball_jumps) / len(ball_jumps) if ball_jumps else 0.0
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark ball/rim settings")
    parser.add_argument(
        "--source",
        default="assets/videos/video_03_expert_score.mp4",
    )
    parser.add_argument("--max-frames", type=int, default=120)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        print(f"Source not found: {source}")
        return 1

    detector = BallDetector("ball.yaml")
    if not detector.ready:
        print("Ball/rim model is not ready.")
        return 2

    print(
        "name | imgsz | ball hit/conf/gap | rim hit/conf/gap | "
        "ball jump px | FPS"
    )
    for config in CONFIGS:
        stats = run_config(detector, source, config, args.max_frames)
        ball_conf = stats.ball_confidence / max(stats.ball_hits, 1)
        rim_conf = stats.rim_confidence / max(stats.rim_hits, 1)
        print(
            f"{config.name} | {config.imgsz} | "
            f"{stats.ball_hits}/{stats.frames} {ball_conf:.3f} "
            f"gap={stats.ball_max_gap} | "
            f"{stats.rim_hits}/{stats.frames} {rim_conf:.3f} "
            f"gap={stats.rim_max_gap} | "
            f"{stats.ball_mean_jump:.1f} | {stats.fps:.1f}"
        )

    print(
        "\nCoverage is not accuracy. Use labeled ball/rim boxes to calculate "
        "precision, recall, and mAP before choosing production thresholds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
