"""Headless smoke test for the integrated pose + ball/rim pipeline.

Usage:
    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --max-frames 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from mediapipe.tasks.python import vision

from pipeline import ShotAnalysisPipeline
from pose.detector import PoseDetector
from utils.timestamps import frame_timestamp_ms
from visualization.renderer import render_frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify integrated pose + ball/rim processing"
    )
    parser.add_argument(
        "--source",
        default="assets/videos/video_03_expert_score.mp4",
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument(
        "--output",
        default="outputs/ball_rim_verify/pipeline_integrated.mp4",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        print(f"Source not found: {source}")
        return 1

    pose_detector = PoseDetector(running_mode=vision.RunningMode.VIDEO)
    pipeline = ShotAnalysisPipeline()
    if not pipeline.ball_enabled:
        print("Ball/rim detector is not enabled.")
        return 2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"Could not open video: {source}")
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline.set_fps(fps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    frames = pose_frames = ball_frames = rim_frames = 0
    started = time.perf_counter()

    while frames < args.max_frames:
        ok, bgr_frame = capture.read()
        if not ok:
            break

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = frame_timestamp_ms(frames, fps)
        pose_result = pose_detector.detect_video_frame(rgb_frame, timestamp_ms)
        height, width = bgr_frame.shape[:2]
        result = pipeline.process_frame(
            pose_result,
            width,
            height,
            timestamp_ms,
            bgr_frame=bgr_frame,
        )

        pose_frames += int(result.has_pose)
        ball_frames += int(result.ball is not None)
        rim_frames += int(result.rim is not None)

        annotated_rgb = render_frame(rgb_frame, pose_result, result)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height),
            )
        writer.write(annotated_bgr)
        frames += 1

    capture.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - started
    print("--- integrated pipeline summary ---")
    print(f"device:      {pipeline.ball_device}")
    print(f"frames:      {frames}")
    print(f"pose frames: {pose_frames}")
    print(f"ball frames: {ball_frames}")
    print(f"rim frames:  {rim_frames}")
    print(f"ball buffer: {len(pipeline.ball_buffer)}")
    print(f"throughput:  {frames / max(elapsed, 1e-6):.1f} FPS")
    print(f"saved:       {output_path}")

    if frames == 0 or pose_frames == 0 or ball_frames == 0 or rim_frames == 0:
        print("ERROR: one or more integrated detection branches produced no results.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
