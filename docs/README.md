# Swichy Documentation Index

Read these in order to learn the system and complete remaining work manually.

---

## Start here

| # | Document | What you will learn |
|---|----------|---------------------|
| **0** | **[MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md)** | **Study path, file map, what's done vs stubbed** |
| 1 | [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | All phases 1→6 — implementation map |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | Module design, separation of concerns |
| 3 | [PIPELINE.md](PIPELINE.md) | End-to-end data flow, `FrameResult` |
| 4 | [ANGLES_3D.md](ANGLES_3D.md) | 3D vector math, joint chains, index finger |
| 5 | [FILTERS.md](FILTERS.md) | One Euro Filter |
| 6 | [VISIBILITY.md](VISIBILITY.md) | Occlusion gating + report gap notes |
| 7 | [PHASE_DETECTION.md](PHASE_DETECTION.md) | 8-phase FSM, index finger signals |
| 8 | [BIOMECHANICS.md](BIOMECHANICS.md) | Rule engine (12 rules) |
| 9 | [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | Research papers → rule rationale |
| 10 | [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md) | Shot score, mid-entry, performance plan |
| 11 | [REPORTING.md](REPORTING.md) | PDF reports, drills, capture status |
| 12 | [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | **Next build:** ball + make/miss |
| 13 | [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Commercial roadmap (#1–#17) |

---

## Development status

| Phase | Status | Key modules |
|-------|--------|-------------|
| 1 — Foundation | **Done** | `pose/`, `geometry/`, `angles/`, `pipeline.py` |
| 2 — Filter + visibility | **Done** | `filters/`, `pose/visibility.py` |
| 3 — Phase detection | **Done** | `phase_detection/`, index finger in FSM |
| 4 — Biomechanical rules | **Done** | `analysis/`, `config/biomechanics.yaml` |
| 5 — Scoring + coaching | **Done** | `feedback/shot_tracker.py`, `scorer.py` |
| 5b — PDF reports | **Done** | `feedback/report_pdf.py`, `performance_plan.py` |
| 6 — Ball + outcome | **Stubs** | `ball/` — implement per PHASE_6 doc |
| 11 — Physics trajectory | **Stubs** | `physics/` |

---

## Full pipeline (current)

```
Camera → Pose → Filter → Visibility → 3D Angles → Features → Phases → Rules
  → Shot Tracker → Score → Performance Plan → HUD + PDF Report
```

Entry point: [`pipeline.py`](../pipeline.py) → `ShotAnalysisPipeline.process_frame()`

---

## Test media

See [`assets/README.md`](../assets/README.md) for front / side / back view clips and dunk vs jump-shot labels.

---

## Run tests

```powershell
.\venv\Scripts\activate
python tests/test_angles.py
python tests/test_phases.py
python tests/test_feedback.py
python tests/test_performance_plan.py
python tests/test_visibility_gaps.py
python tests/test_hud_display.py
python tests/test_reporting.py
```
