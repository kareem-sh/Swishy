# Swichy — AI Basketball Shooting Coach

Real-time basketball shooting analysis using MediaPipe Pose, 3D vector biomechanics, phase detection, and PDF performance reports.

## Features (Implemented)

- Full-body pose detection (MediaPipe Pose Landmarker **full** model, CPU on Windows)
- **3D world-space joint angles** including **index finger alignment** (elbow → wrist → index)
- Visibility-gated **ankle flexion** from knee → ankle → shoe tip (diagnostic; not scored)
- **One Euro Filter** for real-time landmark smoothing
- **Visibility + presence gating** with temporal hold and gap notes in reports
- **8-phase shot FSM** with hysteresis, dwell time, and set-shot paths
- **12 biomechanical rules** (YAML-driven, research-informed)
- **Per-shot scoring (0–100)** with grade, drills, and action items
- **Mid-shot entry** — detects and scores reps that start mid-video/live with clear warnings
- **PDF session reports** with embedded key frames, practice plan, and capture status
- Structured **HUD** with smoothed on-screen text (`visualization/hud_display.py`)
- Custom **basketball + rim YOLO** detection with CUDA auto-selection
- Ball tracking, rim persistence, and overlays in image/video/live modes
- Live webcam, video file, and image modes
- Test assets: front / side / back view jump shots, dunks, layups

## In Progress

- **Phase 6 outcome:** release synchronization and validated make/miss reporting
  → [`ball/`](ball/), [`docs/PHASE_6_BALL_AND_OUTCOME.md`](docs/PHASE_6_BALL_AND_OUTCOME.md)
- **Physics trajectory (#11):** → [`physics/`](physics/)

---

## Quick Start

### Windows

```powershell
cd C:\Users\karim\Desktop\Swichy
.\venv\Scripts\Activate.ps1

# NVIDIA GPU: install CUDA PyTorch first (see docs/GPU_YOLO_SETUP.md)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

# Download model once (skip if models/pose_landmarker_full.task exists)
New-Item -ItemType Directory -Force -Path models | Out-Null
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models\pose_landmarker_full.task"

python main.py
```

**Use the venv Python** after `activate` — not Anaconda's global Python.
For GPU verification and model details, read
[docs/GPU_YOLO_SETUP.md](docs/GPU_YOLO_SETUP.md).

### Change mode in [`main.py`](main.py)

```python
MODE = "video"   # live | image | video

# Side-view jump shot (best for biomechanics):
run_video_mode("assets/videos/video_07_side_jump_shot.mp4")
```

Reports save to `outputs/reports/{session_id}/REPORT.pdf`.

---

## Project Structure

```
Swichy/
├── main.py                 # Entry point
├── pipeline.py             # Central analysis pipeline
├── config/                 # YAML + settings.py
├── pose/                   # Detection, landmarks, visibility
├── filters/                # One Euro Filter
├── geometry/ + angles/     # 3D joint math
├── phase_detection/        # 8-phase FSM
├── analysis/               # Biomechanical rule engine
├── feedback/               # Scoring, coaching, PDF reports
├── visualization/          # Renderer + HUD
├── modes/                  # live, video, image
├── ball/                   # Ball/rim detection, tracking, outcome experiments
├── physics/                # Trajectory stubs
├── assets/                 # Test media
├── docs/                   # Full documentation
└── tests/
```

---

## Documentation — Start Here

| Doc                                                | Purpose                                    |
| -------------------------------------------------- | ------------------------------------------ |
| **[docs/README.md](docs/README.md)**               | **Ordered beginner learning path**         |
| [docs/PIPELINE.md](docs/PIPELINE.md)               | Current pose + ball/rim data flow          |
| [docs/LANDMARKS.md](docs/LANDMARKS.md)             | Index finger, feet, shoes, and reliability |
| [docs/PHASE_DETECTION.md](docs/PHASE_DETECTION.md) | Features, FSM, hysteresis, tuning          |
| [docs/GPU_YOLO_SETUP.md](docs/GPU_YOLO_SETUP.md)   | CUDA and YOLO setup                        |
| [docs/PRODUCT_ROADMAP.md](docs/PRODUCT_ROADMAP.md) | Mobile app, datasets, team tasks           |
| [assets/README.md](assets/README.md)               | Test videos by camera angle                |

The full index keeps filenames stable and gives the exact recommended order.

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
python scripts/check_docs.py
```

---

## Tech Stack

- Python 3.10+
- MediaPipe Pose Landmarker
- Ultralytics + CUDA-enabled PyTorch (basketball/rim YOLO)
- OpenCV
- NumPy, PyYAML
- fpdf2 (PDF reports)
- yt-dlp (optional — re-download YouTube test clip)

COURT DETECTION: Detect court and player position in 3D using KaliCalip hugging face model

python -m court.detector assets/images/court/side.png \
 --device cpu --output assets/images/keypoints/side_check.png \
 --print-court-coordinates --point 500,600
