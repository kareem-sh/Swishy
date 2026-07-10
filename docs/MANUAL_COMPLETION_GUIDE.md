# Manual Study & Completion Guide

Use this document to **learn the Swichy codebase** and **finish the remaining work yourself**. Everything implemented today is documented here with file paths, reading order, and what is still stubbed.

---

## What Swichy Does Today (Complete)

```
Camera / Video / Image
  → Pose (MediaPipe)
  → One Euro Filter
  → Visibility + Presence Gating
  → 3D Joint Angles (+ index finger alignment)
  → Phase FSM (8 phases, hysteresis + dwell)
  → Biomechanical Rules (12 rules, YAML-driven)
  → Shot Scoring (0–100) + Coaching Tips
  → PDF Session Report (drills, action items, key frames)
```

**Entry point:** [`main.py`](../main.py) → [`modes/`](../modes/) → [`pipeline.py`](../pipeline.py)

---

## Recommended Reading Order (Study the Code)

| Step | Read | Then open in code |
|------|------|-------------------|
| 1 | [ARCHITECTURE.md](ARCHITECTURE.md) | Folder structure |
| 2 | [PIPELINE.md](PIPELINE.md) | `pipeline.py` — trace `process_frame()` |
| 3 | [ANGLES_3D.md](ANGLES_3D.md) | `angles/calculator.py`, `geometry/vectors.py` |
| 4 | [FILTERS.md](FILTERS.md) + [VISIBILITY.md](VISIBILITY.md) | `filters/one_euro.py`, `pose/visibility.py` |
| 5 | [PHASE_DETECTION.md](PHASE_DETECTION.md) | `phase_detection/detector.py`, `features.py` |
| 6 | [BIOMECHANICS.md](BIOMECHANICS.md) + [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | `analysis/engine.py`, `config/biomechanics.yaml` |
| 7 | [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md) | `feedback/shot_tracker.py`, `scorer.py` |
| 8 | [REPORTING.md](REPORTING.md) | `feedback/report_pdf.py`, `session_recorder.py` |
| 9 | [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | `ball/` stubs — **your next build** |
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
│   ├── joint_chains.py          # elbow, knee, hip, shoulder, index_align, trunk
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
├── ball/                        # ⚠ STUBS — Phase 6 (you implement)
├── physics/                     # ⚠ STUBS — Improvement #11
│
├── config/
│   ├── settings.py              # Thresholds, paths, shooting hand
│   ├── filter_config.yaml
│   ├── phases.yaml              # FSM thresholds
│   ├── biomechanics.yaml        # 12 rules
│   ├── scoring.yaml
│   ├── display.yaml             # HUD hold frames, video playback speed
│   ├── report_config.yaml
│   ├── ball.yaml                # Phase 6 config (stub)
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
```

---

## What You Still Need to Build (Manual Completion)

### Priority 1 — Phase 6: Ball & Shot Outcome
**Plan:** [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md)

| Stub file | Your task |
|-----------|-----------|
| `ball/detector.py` | Detect ball per frame (color/YOLO) |
| `ball/tracker.py` | Track ball across frames |
| `ball/timeseries.py` | Ball height/velocity series |
| `ball/release_sync.py` | Align ball release with pose release phase |
| `ball/trajectory.py` | Fit arc |
| `ball/outcome.py` | Make/miss classification |
| `ball/fusion.py` | Merge ball + pose into `ShotSummary` |

**Integration point:** Call from `pipeline.py` after phase detection; extend `ShotSummary` with outcome field.

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
| 4 | Start Phase 6 — ball detection on still frames |
| 5 | Ball tracking + timeseries |
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
| [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | Phases 1–6 summary |
| [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) | Papers → rules mapping |
| [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md) | Ball tracking design |
| [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | Long-term roadmap |
