"""Save annotated key frames and write markdown session reports."""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from feedback.report_models import DetailedShotReport, SessionReport
from visualization.report_frame import annotate_key_frame


def write_session_report(
    report: SessionReport,
    frame_images: Dict[int, np.ndarray],
    output_dir: Path,
) -> Path:
    session_dir = output_dir / report.session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_images: Dict[str, np.ndarray] = {}

    for shot in report.shots:
        for kf in shot.key_frames:
            base_img = frame_images.get(kf.frame_index)
            if base_img is None:
                continue
            annotated = annotate_key_frame(base_img, kf)
            out_path = frames_dir / kf.image_filename
            cv2.imwrite(str(out_path), annotated)
            saved_images[kf.image_filename] = annotated

            for ev in shot.rule_evaluations:
                if ev.rule_id in kf.rule_ids:
                    ev.key_frame_filename = kf.image_filename

    report_path = session_dir / "REPORT.md"
    report_path.write_text(_render_markdown(report), encoding="utf-8")
    return report_path


def _render_markdown(report: SessionReport) -> str:
    lines = [
        "# Swichy — Shooting Form Analysis Report",
        "",
        "## Session Overview",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Source | `{report.source_name}` |",
        f"| Type | {report.source_type} |",
        f"| Session ID | `{report.session_id}` |",
        f"| FPS | {report.fps:.1f} |",
        f"| Frames analyzed | {report.total_frames} |",
        f"| Shots detected | {report.shot_count} |",
        f"| Overall score | **{report.overall_score}/100** ({report.overall_grade}) |",
        "",
    ]

    if report.strengths:
        lines += ["## Strengths", ""]
        for s in report.strengths:
            lines.append(f"- {s}")
        lines.append("")

    if report.top_improvements:
        lines += ["## Priority Improvements", ""]
        for i, item in enumerate(report.top_improvements, 1):
            lines.append(f"{i}. **{item}**")
        lines.append("")

    if not report.shots:
        lines += [
            "## No Shots Detected",
            "",
            "The system did not detect a complete shot cycle (loading → landing).",
            "Ensure the player is fully visible and performs a jump shot.",
            "",
        ]
        return "\n".join(lines)

    lines += ["---", "", "## Shot-by-Shot Analysis", ""]

    for shot in report.shots:
        s = shot.summary
        lines += [
            f"### Shot #{s.shot_number} — {s.grade} ({s.score}/100)",
            "",
            f"- **Time:** {_ts(shot.start_timestamp_ms)} → {_ts(shot.end_timestamp_ms)}",
            f"- **Rules passed:** {s.passed_count}/{s.total_count}",
            f"- **Phases detected:** {', '.join(s.phases_seen) or 'N/A'}",
            "",
        ]

        if s.coaching_tips:
            lines += ["#### Coach Summary", ""]
            for tip in s.coaching_tips:
                lines.append(f"- {tip}")
            lines.append("")

        if shot.phase_timeline:
            lines += [
                "#### Phase Timeline",
                "",
                "| Time | Phase | Frame |",
                "|------|-------|-------|",
            ]
            for pm in shot.phase_timeline:
                lines.append(f"| {pm.timestamp_label} | {pm.phase_label} | {pm.frame_index} |")
            lines.append("")

        lines += ["#### Form Checklist", ""]
        for ev in shot.rule_evaluations:
            icon = "PASS" if ev.passed else "FAIL"
            measured = f"{ev.measured_value:.1f}" if ev.measured_value is not None else "N/A"
            range_str = _range_str(ev.min_value, ev.max_value)
            lines.append(f"- **[{icon}]** {ev.name} — measured `{measured}` (range {range_str})")
            lines.append(f"  - {ev.message}")
            if ev.rationale:
                lines.append(f"  - *Why:* {ev.rationale}")
            if ev.key_frame_filename:
                lines.append(f"  - See frame: `frames/{ev.key_frame_filename}`")
            lines.append("")

        if shot.key_frames:
            lines += ["#### Key Frames — Where to Improve", ""]
            for kf in shot.key_frames:
                lines += [
                    f"##### {kf.capture_reason} @ {kf.timestamp_label}",
                    "",
                    f"![{kf.capture_reason}](frames/{kf.image_filename})",
                    "",
                    f"- **Phase:** {kf.phase_label}",
                    f"- **Frame:** {kf.frame_index}",
                ]
                if kf.angles_summary:
                    lines.append(f"- **Angles:** {kf.angles_summary}")
                if kf.issues:
                    lines.append("- **Issues on this frame:**")
                    for issue in kf.issues:
                        lines.append(f"  - {issue}")
                lines.append("")

        lines += ["---", ""]

    lines += [
        "## How to Use This Report",
        "",
        "1. Start with **Priority Improvements** — fix the most frequent issues first.",
        "2. Open **Key Frames** — each image shows exactly when and where form broke down.",
        "3. Use the **Phase Timeline** to understand shot sequencing.",
        "4. Re-record and compare overall score across sessions.",
        "",
        "*Generated by Swichy AI Basketball Coach*",
    ]
    return "\n".join(lines)


def _ts(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _range_str(min_v, max_v) -> str:
    if min_v is not None and max_v is not None:
        return f"{min_v}–{max_v}"
    if min_v is not None:
        return f"≥ {min_v}"
    if max_v is not None:
        return f"≤ {max_v}"
    return "N/A"
