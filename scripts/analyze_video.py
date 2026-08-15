"""CLI entry point for headless video analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feedback.console import save_session_report_json
from modes.headless_video import analyze_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a basketball video headlessly")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--height-cm",
        type=float,
        default=None,
        help="Player height supplied by the client, in centimetres",
    )
    parser.add_argument(
        "--rim-height-m",
        type=float,
        default=None,
        help="Rim height supplied by the client, in metres",
    )
    parser.add_argument(
        "--shot-xy-m",
        type=float,
        nargs=2,
        metavar=("X_M", "Y_M"),
        default=None,
        help="FIBA half-court shot coordinates in metres",
    )
    parser.add_argument(
        "--json-out",
        help="Optional path to write session JSON report",
    )
    parser.add_argument(
        "--key-frames-dir",
        help="Optional directory to save per-shot key frame images",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Also write Swishy PDF report to outputs/reports",
    )
    args = parser.parse_args()

    payload = analyze_video(
        args.video,
        height_cm=args.height_cm,
        rim_height_m=args.rim_height_m,
        shot_xy_m=args.shot_xy_m,
        save_key_frames=bool(args.key_frames_dir),
        key_frames_dir=args.key_frames_dir,
        save_report=args.save_report,
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Session JSON saved: {out_path}")

    name = payload.get("video") or payload.get("source_name")
    score = payload.get("average_score")
    if score is None:
        score = payload.get("overall_score")
    print(
        f"Analyzed {name}: "
        f"{payload['shot_count']} shot(s), overall score {score}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
