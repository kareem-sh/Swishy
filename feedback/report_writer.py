"""Save annotated key frames and write PDF session reports."""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from feedback.report_models import DetailedShotReport, SessionReport
from feedback.report_pdf import write_session_pdf
from utils.config_loader import load_yaml
from visualization.report_frame import annotate_key_frame


def write_session_report(
    report: SessionReport,
    frame_images: Dict[int, np.ndarray],
    output_dir: Path,
) -> Path:
    cfg = load_yaml("report_config.yaml")
    save_markdown = bool(cfg.get("save_markdown_copy", False))

    session_dir = output_dir / report.session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for shot in report.shots:
        for kf in shot.key_frames:
            base_img = frame_images.get(kf.frame_index)
            if base_img is None:
                continue
            annotated = annotate_key_frame(base_img, kf)
            out_path = frames_dir / kf.image_filename
            cv2.imwrite(str(out_path), annotated)

            for ev in shot.rule_evaluations:
                if ev.rule_id in kf.rule_ids:
                    ev.key_frame_filename = kf.image_filename

    pdf_path = session_dir / "REPORT.pdf"
    write_session_pdf(report, frames_dir, pdf_path)

    if save_markdown:
        md_path = session_dir / "REPORT.md"
        md_path.write_text(_render_markdown(report), encoding="utf-8")

    return pdf_path


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
            "",
        ]
        return "\n".join(lines)

    for shot in report.shots:
        s = shot.summary
        lines += [
            f"### Shot #{s.shot_number} — {s.grade} ({s.score}/100)",
            "",
            f"- Rules passed: {s.passed_count}/{s.total_count}",
        ]
        if s.outcome is not None:
            lines.append(
                f"- Basket outcome: {s.outcome.result.upper()} "
                f"({s.outcome.confidence:.0%} confidence)"
            )
        lines.append("")
        for ev in shot.rule_evaluations:
            icon = "PASS" if ev.passed else "FAIL"
            lines.append(f"- [{icon}] {ev.name}: {ev.message}")
        lines.append("")

    return "\n".join(lines)
