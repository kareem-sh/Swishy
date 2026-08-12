# Manual Study & Completion Guide

Use this document to practise working through the Swichy codebase and finish
the remaining experimental features. For the canonical reading order and
implementation status, start with [docs/README.md](README.md).

---

## What Swichy Does Today (Complete)

```text
Camera / Video / Image
├─ BGR → basketball YOLO → ball/rim → temporal tracker
└─ RGB → MediaPipe → filter → visibility → 3D joint angles
                       → phase FSM → rules → score → PDF report
```

**Entry point:** [`main.py`](../main.py) → [`modes/`](../modes/) → [`pipeline.py`](../pipeline.py)

---

## Quick code map

The canonical sequence is [docs/README.md](README.md); this table only maps
topics to source files.

| Step | Read | Then open in code |
|------|------|-------------------|
| 1 | [ARCHITECTURE.md](ARCHITECTURE.md) | Folder structure |
| 2 | [PIPELINE.md](PIPELINE.md) | `pipeline.py` — trace `process_frame()` |
| 3 | [LANDMARKS.md](LANDMARKS.md) + [ANGLES_3D.md](ANGLES_3D.md) | `pose/landmarks.py`, `angles/calculator.py` |
| 4 | [FILTERS.md](FILTERS.md) + [VISIBILITY.md](VISIBILITY.md) | `filters/one_euro.py`, `pose/visibility.py` |
| 5 | [PHASE_DETECTION.md](PHASE_DETECTION.md) | `phase_detection/detector.py`, `features.py` |
| 6 | [BIOMECHANICS.md](BIOMECHANICS.md) + [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | `analysis/engine.py`, `config/biomechanics.yaml` |
| 7 | [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md) | `feedback/shot_tracker.py`, `scorer.py` |
| 8 | [REPORTING.md](REPORTING.md) | `feedback/report_pdf.py`, `session_recorder.py` |
| 9 | [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | Integrated detection; outcome fusion remains |
| 10 | [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md) | **Mobile app, datasets, team tasks, timeline** |
| 11 | [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Roadmap after Phase 6 |

**Hands-on:** Run `python main.py` with `MODE = "video"` and a test clip from [`assets/README.md`](../assets/README.md). Set breakpoints in `pipeline.py` line 120 (`process_frame`).

---

## Full File Map (What Each Module Does)

```
Swichy/
├── main.py                      # MODE switch: live | video | image
├── pipeline.py                  # ShotAnalysisPipeline — orchestrates everything
│
├── pose/
│   ├── detector.py              # MediaPipe Pose Landmarker wrapper
│   ├── landmarks.py             # Image + world landmarks, index finger IDs
│   └── visibility.py            # min(visibility, presence) gating + hold
│
├── filters/
│   └── one_euro.py              # Per-landmark adaptive smoothing
│
├── geometry/
│   └── vectors.py               # 3D angle math (dot product, vertical)
│
├── angles/
│   ├── joint_chains.py          # body, index alignment, ankle-flexion chains
│   └── calculator.py            # AngleResult per joint
│
├── phase_detection/
│   ├── phases.py                # PHASE_ORDER, TRANSITIONS, labels
│   ├── features.py              # Velocities, index finger kinematics
│   └── detector.py              # FSM with hysteresis + min_dwell_frames
│
├── analysis/
│   ├── engine.py                # BiomechanicsEngine — YAML rules
│   └── models.py                # RuleResult, AnalysisResult
│
├── feedback/
│   ├── shot_tracker.py          # Shot boundaries + mid-entry detection
│   ├── scorer.py                # Weighted 0–100 aggregation
│   ├── generator.py             # Coaching tips text
│   ├── performance_plan.py      # Drills, action items, capture notes
│   ├── visibility_gaps.py       # Report when landmarks occluded too long
│   ├── session_recorder.py      # Collects shots for reports
│   ├── frame_capture.py         # Picks key frames per shot
│   ├── report_builder.py        # DetailedShotReport + SessionReport
│   ├── report_pdf.py            # PDF output (primary)
│   ├── report_writer.py         # Saves PDF + optional markdown
│   └── console.py               # Terminal shot summary
│
├── visualization/
│   ├── renderer.py              # Skeleton + delegates to HUD
│   ├── hud.py                   # Panels: phase, angles, violations, summary
│   └── hud_display.py           # Temporal smoothing so text is readable
│
├── modes/
│   ├── live_stream.py           # Webcam
│   ├── video_mode.py            # File playback + SessionRecorder
│   └── image_mode.py            # Single-frame analysis
│
├── ball/                        # Phase 6a integrated; outcome fusion experimental
├── physics/                     # Experimental improvement #11
│
├── config/
│   ├── settings.py              # Thresholds, paths, shooting hand
│   ├── filter_config.yaml
│   ├── phases.yaml              # FSM thresholds
│   ├── biomechanics.yaml        # 12 rules
│   ├── scoring.yaml
│   ├── display.yaml             # HUD hold frames, video playback speed
│   ├── report_config.yaml
│   ├── ball.yaml                # Active YOLO, tracking, and outcome config
│   ├── hoop_roi.yaml
│   └── physics.yaml
│
├── tests/                       # Run each file with python tests/test_*.py
├── assets/                      # Test videos + images
├── docs/                        # You are here
└── outputs/reports/             # Generated PDFs (gitignored)
```

---

## Key Ideas to Understand Before Extending

### 1. World landmarks vs image landmarks
- **Image** (`pose_landmarks`) → drawing skeleton on screen only.
- **World** (`pose_world_landmarks`) → all angle math. Meters, hip-centered.

### 2. Phase-gated rules
A bent elbow is correct during **loading** and wrong at **release**. Rules in `biomechanics.yaml` list which `phases` they apply to.

### 3. FSM stability
`config/phases.yaml`:
- `hysteresis_frames: 5` — need 5 agreeing frames before switching phase.
- `min_dwell_frames: 3` — stay in a phase at least 3 frames before leaving.
Prevents flicker when the player pauses mid-motion.

### 4. Mid-shot entry
If video/live starts during **Jump** or **Release**, `ShotTracker` still records the rep and flags `started_mid_phase=True`. PDF reports explain what was not captured. See `feedback/shot_tracker.py`.

### 5. Performance-oriented PDF
Reports are not just scores — they include **Next Rep Focus**, **Drills**, **Action Items**, and **Practice Plan**. See `feedback/performance_plan.py`.

---

## Configuration Cheat Sheet

| Want to change… | Edit |
|-----------------|------|
| Smoothing vs responsiveness | `config/filter_config.yaml` |
| Phase sensitivity | `config/phases.yaml` |
| Coaching rules & ranges | `config/biomechanics.yaml` |
| Score weights | `config/scoring.yaml` |
| HUD text speed / video playback | `config/display.yaml` |
| Report key-frame count | `config/report_config.yaml` |
| Shooting hand | `config/settings.py` → `SHOOTING_HAND` |

---

## Test Assets

| Clip | Use for |
|------|---------|
| `assets/test.mp4` | Front-view jump shot (default) |
| `assets/videos/video_07_side_jump_shot.mp4` | Side view — best for biomechanics |
| `assets/videos/video_05_pair_training.mp4` | Back view |
| `assets/videos/video_03/04/06_*.mp4` | Dunks (not jump-shot form) |

Full table: [`assets/README.md`](../assets/README.md)

---

## Run Tests

```powershell
.\venv\Scripts\activate
python tests/test_angles.py
python tests/test_phases.py
python tests/test_feedback.py
python tests/test_performance_plan.py
python tests/test_visibility_gaps.py
python tests/test_hud_display.py
python tests/test_reporting.py
python tests/test_ball_tracking.py
python tests/test_runtime_utilities.py
```

---

## What You Still Need to Build (Manual Completion)

### Priority 1 — Complete Phase 6 outcome fusion
**Plan:** [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md)

| Module | Current state and next task |
|-----------|-----------|
| `ball/detector.py` | Integrated; validate precision/recall on labeled phone footage |
| `ball/tracker.py` | Integrated; tune short-gap prediction on real shots |
| `ball/timeseries.py` | Integrated; preserve per-shot evidence for reports |
| `ball/release_sync.py` | Experimental; validate ball–wrist release alignment |
| `ball/trajectory.py` | Experimental; validate arc fit and entry-angle proxy |
| `ball/outcome.py` | Experimental; classify completed shot windows |
| `ball/fusion.py` | Experimental; attach outcome to `ShotSummary` and reports |

**Integration point:** Ball detection already runs in `pipeline.py`. Remaining
work starts when a shot window completes: classify its ball time series, attach
the result to `ShotSummary`, and pass it to session reporting.

### Priority 2 — Physics trajectory (#11)
**Stubs:** `physics/trajectory.py`, `physics/models.py`  
Uses ball release point + velocity estimate to predict arc.

### Priority 3 — Product polish (pick any)
From [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md):
- Coach dashboard (#16)
- Personalized angle ranges per player (#5)
- Shot type classification — dunk vs jump shot vs layup (#8)
- ML-based phases (#2, #3) if you collect labeled data

---

## Suggested Weekly Plan (Self-Study)

| Week | Goal |
|------|------|
| 1 | Run app, read pipeline + angles, tune one rule in `biomechanics.yaml` |
| 2 | Understand FSM — tune `phases.yaml` on side-view video |
| 3 | Read scoring + reports — generate PDFs, verify drills make sense |
| 4 | Validate ball/rim boxes and label representative phone footage |
| 5 | Fine-tune detection or tracking only where measurements show a gap |
| 6 | Release sync + make/miss outcome |
| 7 | Integrate outcome into PDF report |
| 8+ | Dashboard, mobile, or ML experiments |

---

## Debugging Tips

1. **No pose / N/A angles** — check lighting, full body in frame, `VISIBILITY_THRESHOLD` in `settings.py`.
2. **Wrong shooting side** — set `SHOOTING_HAND = "left"` or `"right"` in `settings.py`.
3. **Phases flicker** — increase `hysteresis_frames` in `phases.yaml`.
4. **Only one shot detected** — video may start mid-rep; check PDF Capture Status notes.
5. **Wrong Python** — always `.\venv\Scripts\activate` before `python main.py`.

---

## Doc Index

| Document | Topic |
|----------|-------|
| [README.md](README.md) | Doc index |
| [LANDMARKS.md](LANDMARKS.md) | Index finger, feet, shoes, and reliability |
| [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | Phases 1–6 summary |
| [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | Papers → rules mapping |
| [GPU_YOLO_SETUP.md](GPU_YOLO_SETUP.md) | CUDA and detector verification |
| [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | Ball tracking design |
| [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Long-term roadmap |
