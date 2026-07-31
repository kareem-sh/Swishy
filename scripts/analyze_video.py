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

    print(
        f"Analyzed {payload['source_name']}: "
        f"{payload['shot_count']} shot(s), overall score {payload['overall_score']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
