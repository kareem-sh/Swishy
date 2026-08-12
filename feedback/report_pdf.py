"""Generate PDF session reports with embedded key-frame images."""

from pathlib import Path

from fpdf import FPDF

from feedback.report_models import DetailedShotReport, SessionReport


class _ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Swichy AI Basketball Coach  |  Page {self.page_no()}", align="C")


def write_session_pdf(
    report: SessionReport,
    frames_dir: Path,
    output_path: Path,
) -> Path:
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    _add_title_page(pdf, report)

    if not report.shots:
        pdf.add_page()
        pdf.set_font("Helvetica", "", 11)
        _multi(pdf,
            "No complete shots were detected. Ensure the full body is visible "
            "and perform a full jump shot cycle (loading through landing)."
        )
    else:
        for shot in report.shots:
            _add_shot_section(pdf, shot, frames_dir)

    pdf.output(str(output_path))
    return output_path


def _safe(text: str) -> str:
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2192", "->")


def _multi(pdf: FPDF, text: str, h: float = 5, bold: bool = False):
    pdf.set_x(pdf.l_margin)
    if bold:
        pdf.set_font("Helvetica", "B", pdf.font_size_pt)
    pdf.multi_cell(pdf.epw, h, _safe(text))
    if bold:
        pdf.set_font("Helvetica", "", pdf.font_size_pt)


def _add_title_page(pdf: FPDF, report: SessionReport):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 14, "Swichy Shooting Form Report", ln=True)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.ln(4)
    pdf.cell(0, 7, f"Source: {report.source_name}", ln=True)
    pdf.cell(0, 7, f"Type: {report.source_type}  |  Session: {report.session_id}", ln=True)
    pdf.cell(0, 7, f"FPS: {report.fps:.1f}  |  Frames: {report.total_frames}  |  Shots: {report.shot_count}", ln=True)
    pdf.cell(0, 7, f"Overall score: {report.overall_score}/100 ({report.overall_grade})", ln=True)
    pdf.ln(6)

    if report.strengths:
        _section_heading(pdf, "Strengths")
        for item in report.strengths:
            _multi(pdf, f"  + {item}", 6)
        pdf.ln(2)

    if report.top_improvements:
        _section_heading(pdf, "Priority Improvements")
        for i, item in enumerate(report.top_improvements, 1):
            _multi(pdf, f"  {i}. {item}", 6)
        pdf.ln(2)

    if report.session_notes:
        _section_heading(pdf, "Capture Notes")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(140, 90, 0)
        for note in report.session_notes:
            _multi(pdf, f"  ! {note}", 6)
        pdf.set_text_color(60, 60, 60)
        pdf.ln(2)

    if report.practice_plan:
        _section_heading(pdf, "Your Practice Plan")
        pdf.set_font("Helvetica", "", 10)
        for i, item in enumerate(report.practice_plan, 1):
            _multi(pdf, f"  {i}. {item}", 6)
        pdf.ln(4)

    _section_heading(pdf, "How to Use This Report")
    pdf.set_font("Helvetica", "", 10)
    _multi(pdf,
        "1. Read Priority Improvements and do the Practice Plan drills before your next session. "
        "2. For each shot, check Capture Status if recording started late or ended early. "
        "3. Use Next Rep Focus for one correction per attempt. "
        "4. Review key-frame images to see exactly when form broke down."
    )


def _add_shot_section(pdf: FPDF, shot: DetailedShotReport, frames_dir: Path):
    s = shot.summary
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 10, _safe(f"Shot #{s.shot_number}  -  {s.grade} ({s.score}/100)"), ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(70, 70, 70)
    pdf.cell(0, 6, _safe(
        f"Time: {_ts(shot.start_timestamp_ms)} -> {_ts(shot.end_timestamp_ms)}  |  "
        f"Rules: {s.passed_count}/{s.total_count} passed"
    ), ln=True)
    if s.phases_seen:
        pdf.cell(0, 6, _safe(f"Phases captured: {', '.join(s.phases_seen)}"), ln=True)
    pdf.ln(2)

    _add_capture_status(pdf, shot)

    if s.next_rep_focus:
        _section_heading(pdf, "Next Rep Focus")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 80, 140)
        for i, item in enumerate(s.next_rep_focus, 1):
            _multi(pdf, f"  {i}. {item}", 6)
        pdf.set_text_color(70, 70, 70)
        pdf.ln(2)

    if s.practice_drills:
        _section_heading(pdf, "Drills for This Shot")
        for drill in s.practice_drills:
            _multi(pdf, f"  > {drill}", 6)
        pdf.ln(2)

    # An "Action Items" section used to sit here, printing
    # `performance_actions` -- which held the "Next Rep Focus" lines above
    # with a "Fix X:" prefix stuck on the front. The same advice, twice on
    # one page.

    if s.coaching_tips:
        _section_heading(pdf, "Coach Summary")
        for tip in s.coaching_tips:
            _multi(pdf, f"  - {tip}")
        pdf.ln(2)

    if shot.phase_timeline:
        _section_heading(pdf, "Phase Timeline")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(42, 6, "Time")
        pdf.cell(80, 6, "Phase")
        pdf.cell(30, 6, "Frame", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for pm in shot.phase_timeline:
            pdf.cell(42, 5, pm.timestamp_label)
            pdf.cell(80, 5, _safe(pm.phase_label))
            pdf.cell(30, 5, str(pm.frame_index), ln=True)
        pdf.ln(3)

    if shot.visibility_gaps:
        _section_heading(pdf, "Tracking Reliability Notes")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(120, 80, 0)
        _multi(pdf,
            "Some body parts were occluded or below confidence for longer than the "
            "hold window. Angles for those periods were excluded from scoring."
        )
        for gap in shot.visibility_gaps:
            _multi(pdf, f"  ! {gap.summary()}")
        pdf.set_text_color(70, 70, 70)
        pdf.ln(2)

    _section_heading(pdf, "Form Checklist")
    for ev in shot.rule_evaluations:
        status = "PASS" if ev.passed else "FAIL"
        measured = f"{ev.measured_value:.1f}" if ev.measured_value is not None else "N/A"
        range_str = _range_str(ev.min_value, ev.max_value)
        pdf.set_font("Helvetica", "B" if not ev.passed else "", 9)
        color = (40, 120, 40) if ev.passed else (180, 40, 40)
        pdf.set_text_color(*color)
        _multi(pdf, f"[{status}] {ev.name}  -  measured {measured} (range {range_str})")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(70, 70, 70)
        _multi(pdf, f"    {ev.message}")
        if ev.rationale:
            pdf.set_text_color(100, 100, 100)
            _multi(pdf, f"    Why: {ev.rationale}")
        pdf.ln(1)

    if shot.key_frames:
        _section_heading(pdf, "Key Frames - Where to Improve")
        for kf in shot.key_frames:
            _add_key_frame(pdf, kf, frames_dir)


def _add_capture_status(pdf: FPDF, shot: DetailedShotReport):
    s = shot.summary
    if not s.capture_note and not s.missing_phases:
        return

    _section_heading(pdf, "Capture Status")
    pdf.set_font("Helvetica", "", 10)

    if s.started_mid_phase or s.ended_early:
        pdf.set_text_color(160, 100, 0)
    else:
        pdf.set_text_color(40, 120, 60)

    if s.capture_note:
        _multi(pdf, s.capture_note, 6)

    if s.missing_phases:
        from phase_detection.phases import PHASE_LABELS
        labels = ", ".join(PHASE_LABELS.get(p, p) for p in s.missing_phases)
        pdf.set_text_color(100, 100, 100)
        _multi(pdf, f"Phases not scored (not on camera): {labels}", 6)

    if s.started_mid_phase:
        _multi(
            pdf,
            "Swichy still detected your current phase and tracked the shot through "
            "release, follow-through, and landing where visible.",
            6,
        )

    pdf.set_text_color(70, 70, 70)
    pdf.ln(2)


def _add_key_frame(pdf: FPDF, kf, frames_dir: Path):
    img_path = frames_dir / kf.image_filename
    if not img_path.exists():
        return

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 30, 30)
    _multi(pdf, f"{kf.capture_reason} @ {kf.timestamp_label}", 6)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    _multi(pdf, f"Phase: {kf.phase_label}  |  Frame: {kf.frame_index}")
    if kf.angles_summary:
        _multi(pdf, f"Angles: {kf.angles_summary}")

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    if pdf.get_y() > pdf.h - 90:
        pdf.add_page()
    y_before = pdf.get_y()
    try:
        pdf.image(str(img_path), x=pdf.l_margin, w=usable_w)
    except Exception:
        _multi(pdf, "(Image could not be embedded)")
        return
    pdf.set_y(y_before + usable_w * 0.56)

    if kf.issues:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(200, 80, 0)
        pdf.cell(0, 5, "Issues on this frame:", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for issue in kf.issues:
            _multi(pdf, f"  ! {issue}")
    pdf.ln(4)


def _section_heading(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, _safe(title), ln=True)


def _ts(ms: int) -> str:
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def _range_str(min_v, max_v) -> str:
    if min_v is not None and max_v is not None:
        return f"{min_v}-{max_v}"
    if min_v is not None:
        return f">= {min_v}"
    if max_v is not None:
        return f"<= {max_v}"
    return "N/A"
