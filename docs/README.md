# Swichy Documentation

This page is the single source of truth for reading order and implementation
status. Documents keep descriptive filenames so links remain stable; follow
this sequence instead of reading the folder alphabetically.

## Beginner learning path

| # | Document | What you will learn |
|---:|----------|---------------------|
|---|----------|---------------------|
| 1 | [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | What each product phase solves |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) | Module boundaries and ownership |
| 3 | [PIPELINE.md](PIPELINE.md) | How one frame moves through the app |
| 4 | [LANDMARKS.md](LANDMARKS.md) | Index-finger flow, shoe-tip landmarks, and reliability limits |
| 5 | [ANGLES_3D.md](ANGLES_3D.md) | Vectors and visibility-gated 3D joint angles |
| 6 | [FILTERS.md](FILTERS.md) | Why raw landmarks need smoothing |
| 7 | [VISIBILITY.md](VISIBILITY.md) | Occlusion and unreliable landmark handling |
| 8 | [PHASE_DETECTION.md](PHASE_DETECTION.md) | Features, FSM transitions, hysteresis, and tuning |
| 9 | [BIOMECHANICS.md](BIOMECHANICS.md) | Phase-gated form rules |
| 10 | [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | Research behind rule ranges |
| 11 | [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md) | Shot boundaries, scoring, and coaching |
| 12 | [REPORTING.md](REPORTING.md) | Reports and key-frame generation |
| 13 | [GPU_YOLO_SETUP.md](GPU_YOLO_SETUP.md) | CUDA setup and YOLO26 vs custom weights |
| 14 | [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | Ball tracking, rim context, and outcomes |
| 15 | [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md) | Work through remaining tasks manually |
| 16 | [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md) | Datasets, mobile work, and milestones |
| 17 | [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Later-stage product ideas |

## Current implementation status

| Phase | Status | Key modules |
|-------|--------|-------------|
|-------|--------|-------------|
| 1 — Foundation | Done | `pose/`, `geometry/`, `angles/`, `pipeline.py` |
| 2 — Filter + visibility | Done | `filters/`, `pose/visibility.py` |
| 3 — Phase detection | Done | `phase_detection/`, `config/phases.yaml` |
| 4 — Biomechanical rules | Done | `analysis/`, `config/biomechanics.yaml` |
| 5 — Scoring + coaching | Done | `feedback/` |
| 5b — PDF reports | Done | `feedback/report_pdf.py`, `session_recorder.py` |
| 6a — Ball + rim detection | Integrated | custom E-BARD YOLO, tracker, overlays |
| 6b–6d — Release/outcome/trajectory | In progress | modules exist; shot-summary fusion remains |
| Physics trajectory | Experimental | `physics/` |

Pose details:

- Index-tip alignment is integrated into release/follow-through detection,
  biomechanics, scoring, display, and reports.
- Heel and shoe-tip landmarks are extracted and produce a visibility-gated
  ankle-flexion angle.
- Ankle flexion is diagnostic only; it does not affect phases or scores until
  representative footage validates useful thresholds.

## Current frame flow

```text
Video / camera / image
├─ BGR frame → basketball YOLO → ball + rim → ball tracker
└─ RGB frame → MediaPipe pose → filter → visibility → 3D angles
                                  → features → phase FSM → rules
                                  → shot tracker → score/report

Both branches meet in pipeline.FrameResult → renderer/HUD/recorder
```

Entry point: [`pipeline.py`](../pipeline.py) →
`ShotAnalysisPipeline.process_frame()`.

## Practical study loop

1. Read one document.
2. Open the files in its code map.
3. Run the focused test.
4. Change one YAML setting.
5. Re-run the same media and compare output.
6. Restore the setting before moving on.

Test media is catalogued in [`assets/README.md`](../assets/README.md).

## Verification commands

```powershell
.\venv\Scripts\Activate.ps1

python scripts/check_environment.py --require-cuda
python tests/test_angles.py
python tests/test_phases.py
python tests/test_feedback.py
python tests/test_performance_plan.py
python tests/test_visibility_gaps.py
python tests/test_hud_display.py
python tests/test_reporting.py
python tests/test_ball_tracking.py
python tests/test_runtime_utilities.py
python scripts/check_docs.py

python scripts/verify_ball_rim.py --source assets/images/image_03_basketball_shoot.jpg --max-frames 1
python scripts/verify_pipeline.py --max-frames 120
```
