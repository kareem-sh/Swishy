# Swichy — AI Basketball Shooting Coach

Real-time basketball shooting analysis using MediaPipe Pose, 3D vector biomechanics, and adaptive landmark filtering.

## Features (Implemented)

- Full-body pose detection (MediaPipe Pose Landmarker)
- **3D world-space joint angles** (rotation-invariant, not 2D image-plane)
- **One Euro Filter** for real-time landmark smoothing
- **Visibility-aware** angle computation (invalid when occluded)
- **8-phase shot detection** (stance → loading → release → landing)
- **10 configurable biomechanical rules** with phase-aware evaluation
- **Per-shot scoring (0–100)** with grade and coaching tips
- **Detailed session reports** — markdown + annotated key frames (all modes)
- Auto-detect shooting hand (left/right)
- Live webcam, video file, and image modes
- Real-time coaching overlay + console shot report

---

## Quick Start

### Windows (recommended — use the project venv)

```powershell
cd C:\Users\karim\Desktop\Swichy

# Activate the virtual environment
.\venv\Scripts\activate

# Install dependencies (only needed once, or after requirements.txt changes)
python -m pip install -r requirements.txt

# Download MediaPipe model (only needed once — skip if models/pose_landmarker_full.task exists)
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models\pose_landmarker_full.task"

# Run (use the venv python — do NOT use anaconda python.exe directly)
python main.py
```

**Important:** After `.\venv\Scripts\activate`, run `python main.py` — not `C:\Users\karim\anaconda3\python.exe main.py`. Anaconda and the venv are separate; packages installed in one are not visible to the other.

### Linux / macOS

```bash
pip install -r requirements.txt
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
python main.py
```

Change mode in [`main.py`](main.py): `MODE = "live"` | `"image"` | `"video"`

---

## Project Structure

```
Swichy/
├── main.py                 # Entry point
├── pipeline.py             # Central analysis pipeline
├── config/
│   ├── settings.py         # App settings
│   ├── filter_config.yaml  # One Euro parameters
│   ├── phases.yaml         # Phase detection thresholds
│   ├── biomechanics.yaml   # Coaching rules
│   ├── scoring.yaml        # Shot score weights
│   └── report_config.yaml  # Report generation settings
├── phase_detection/        # Phase 3 — shot phase FSM
├── analysis/               # Phase 4 — rule engine
├── feedback/               # Phase 5 — scoring + coaching + reports
├── pose/                   # Detection + landmarks + visibility
├── filters/                # One Euro Filter
├── geometry/               # 3D vector math
├── angles/                 # Joint angle calculator
├── visualization/          # Rendering (no analysis)
├── utils/                  # Timestamps, frame buffer, config loader
├── modes/                  # Live, video, image I/O
├── assets/                 # Test videos and images (included)
├── models/                 # MediaPipe pose model (download once)
├── docs/                   # Engineering + teaching documentation
├── tests/                  # Unit tests
```

---

## Documentation (Start Here)

| Doc | Topic |
|-----|-------|
| **[PHASES_OVERVIEW.md](docs/PHASES_OVERVIEW.md)** | **Phases 1→5 — full implementation guide** |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design and module responsibilities |
| [PIPELINE.md](docs/PIPELINE.md) | End-to-end data flow |
| [ANGLES_3D.md](docs/ANGLES_3D.md) | Phase 1 — 3D angles, vector math |
| [FILTERS.md](docs/FILTERS.md) | Phase 2 — One Euro Filter |
| [VISIBILITY.md](docs/VISIBILITY.md) | Phase 2 — Occlusion handling |
| [PHASE_DETECTION.md](docs/PHASE_DETECTION.md) | Phase 3 — Shot phase FSM |
| [BIOMECHANICS.md](docs/BIOMECHANICS.md) | Phase 4 — Rule engine |
| [FEEDBACK_SCORING.md](docs/FEEDBACK_SCORING.md) | Phase 5 — Shot score + tips |
| [REPORTING.md](docs/REPORTING.md) | Detailed reports + key frames |
| [PHASE_6_BALL_AND_OUTCOME.md](docs/PHASE_6_BALL_AND_OUTCOME.md) | Phase 6 — Ball tracking + make/miss (planned) |
| [FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md) | Commercial roadmap |

---

## Run Tests

```bash
python tests/test_angles.py
python tests/test_phases.py
python tests/test_feedback.py
python tests/test_reporting.py
```

---

## Tech Stack

- Python 3.10+
- MediaPipe Pose Landmarker
- OpenCV
- NumPy
- PyYAML
