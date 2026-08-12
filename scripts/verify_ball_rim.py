"""Verify ball + rim YOLO detection on a sample image or video.

Usage (from repo root):
  python scripts/verify_ball_rim.py
  python scripts/verify_ball_rim.py --source assets/videos/video_03_expert_score.mp4
  python scripts/verify_ball_rim.py --source assets/images/image_03_basketball_shoot.jpg
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

from ball.detector import BallDetector


def draw_detections(frame, court, fps: float | None = None):
    out = frame.copy()
    if court.ball is not None:
        x1, y1, x2, y2 = map(int, court.ball.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.circle(out, (int(court.ball.x), int(court.ball.y)), 4, (0, 165, 255), -1)
        cv2.putText(
            out,
            f"ball {court.ball.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
    if court.rim is not None:
        x1, y1, x2, y2 = map(int, court.rim.bbox_xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out, (int(court.rim.x), int(court.rim.y)), 4, (0, 255, 0), -1)
        cv2.putText(
            out,
            f"rim {court.rim.confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    if fps is not None:
        cv2.putText(
            out,
            f"{fps:.1f} FPS",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ball/rim YOLO detection")
    parser.add_argument(
        "--source",
        default="assets/videos/video_03_expert_score.mp4",
        help="Image or video path",
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride-display", type=int, default=1)
    parser.add_argument("--out-dir", default="outputs/ball_rim_verify")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = BallDetector("ball.yaml")
    if not detector.ready:
        print("YOLO model failed to load.")
        return 1
    print(f"Inference device: {detector.device}")

    ball_hits = 0
    rim_hits = 0
    processed = 0
    t0 = time.perf_counter()

    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"Could not read image: {source}")
            return 1
        court = detector.detect_court(frame, 0, 0)
        processed = 1
        ball_hits = int(court.ball is not None)
        rim_hits = int(court.rim is not None)
        annotated = draw_detections(frame, court)
        out_path = out_dir / f"{source.stem}_detect.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved {out_path}")
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"Could not open video: {source}")
            return 1

        writer = None
        out_video = out_dir / f"{source.stem}_detect.mp4"
        idx = 0
        while idx < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t_frame = time.perf_counter()
            court = detector.detect_court(frame, idx, int(idx * 33))
            dt = time.perf_counter() - t_frame
            fps = 1.0 / dt if dt > 0 else 0.0
            processed += 1
            ball_hits += int(court.ball is not None)
            rim_hits += int(court.rim is not None)

            if idx % args.stride_display == 0:
                annotated = draw_detections(frame, court, fps=fps)
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_video), fourcc, 15.0, (w, h))
                writer.write(annotated)
            idx += 1

        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved {out_video}")

    elapsed = time.perf_counter() - t0
    print("--- detection summary ---")
    print(f"frames: {processed}")
    print(f"ball hits: {ball_hits} ({100.0 * ball_hits / max(processed, 1):.1f}%)")
    print(f"rim hits:  {rim_hits} ({100.0 * rim_hits / max(processed, 1):.1f}%)")
    print(f"elapsed: {elapsed:.2f}s  (~{processed / max(elapsed, 1e-6):.1f} FPS overall)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
