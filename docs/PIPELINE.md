# Analysis Pipeline

## What Changed

| File | Change |
|------|--------|
| [`pipeline.py`](../pipeline.py) | **New** — `ShotAnalysisPipeline` orchestrates all per-frame processing |
| [`utils/frame_buffer.py`](../utils/frame_buffer.py) | **New** — Ring buffer storing recent frame snapshots for temporal analysis |
| [`utils/timestamps.py`](../utils/timestamps.py) | **New** — Stable monotonic timestamps for MediaPipe |
| [`modes/live_stream.py`](../modes/live_stream.py) | **Updated** — Uses pipeline + fixed timestamps |
| [`modes/video_mode.py`](../modes/video_mode.py) | **Updated** — Uses pipeline + `frame_index * 1000/fps` |
| [`modes/image_mode.py`](../modes/image_mode.py) | **Updated** — Uses pipeline |

---

## Why It Changed

Previously, each mode (`live_stream`, `video_mode`, `image_mode`) duplicated the flow: detect → extract → draw. Analysis logic lived inside the drawing function. There was no shared pipeline, no temporal memory, and timestamps were broken in video mode (`time.time()` instead of frame-based).

The pipeline centralizes all processing so every input mode produces identical `FrameResult` output.

---

## Data Flow

```
Camera / Video File / Image
        ↓
┌─────────────────────────────────┐
│  pose/detector.py               │
│  MediaPipe Pose Landmarker      │
│  Output: detection_result       │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  pose/landmarks.py              │
│  extract_all_landmarks()        │
│  Output: image + world dicts    │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  filters/one_euro.py            │
│  LandmarkFilterBank             │
│  Output: smoothed world coords  │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  pose/visibility.py             │
│  VisibilityGate.apply()         │
│  Output: gated + held landmarks │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  angles/calculator.py           │
│  AngleCalculator.compute_all()  │
│  Output: Dict[str, AngleResult] │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  phase_detection/features.py    │
│  extract_features()             │
│  Output: KinematicFeatures      │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  phase_detection/detector.py    │
│  ShotPhaseDetector.update()     │
│  Output: phase (8 states)       │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  analysis/engine.py             │
│  BiomechanicsEngine.evaluate()  │
│  Output: AnalysisResult         │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  feedback/shot_tracker.py       │
│  ShotTracker.update()           │
│  Output: ShotSummary on complete│
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  feedback/scorer.py +           │
│  feedback/generator.py          │
│  Score 0-100 + coaching tips    │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  utils/frame_buffer.py          │
│  FrameBuffer.push()             │
│  Output: temporal history       │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  FrameResult dataclass          │
│  (angles, side, phase, etc.)    │
└─────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│  visualization/renderer.py      │
│  Display skeleton + angles      │
└─────────────────────────────────┘
```

---

## FrameResult

The pipeline output for each frame:

```python
@dataclass
class FrameResult:
    image_landmarks: dict | None
    world_landmarks: dict | None
    angles: dict[str, AngleResult]
    features: KinematicFeatures | None
    analysis: AnalysisResult | None   # Phase 4 rule results
    shooting_side: str
    phase: str                        # e.g. "release"
    phase_label: str                  # e.g. "Release"
    timestamp_ms: int
    has_pose: bool
```

Any consumer (visualization, API, database, mobile app) can use `FrameResult` without depending on OpenCV or MediaPipe.

---

## Usage

```python
from pipeline import ShotAnalysisPipeline

pipeline = ShotAnalysisPipeline(shooting_hand="auto")
pipeline.set_fps(30.0)

frame_result = pipeline.process_frame(
    detection_result=result,
    width=w,
    height=h,
    timestamp_ms=timestamp_ms,
)

if frame_result.has_pose:
  elbow = frame_result.angles["right_elbow"]
  if elbow.is_valid:
      print(f"Elbow: {elbow.degrees:.1f}°")
```

---

## Timestamp Fix

MediaPipe VIDEO and LIVE_STREAM modes require **monotonically increasing** timestamps tied to frame order.

**Before (broken):**
```python
timestamp_ms = int(time.time() * 1000)  # jumps around, breaks tracking
```

**After (correct):**
```python
timestamp_ms = frame_index * 1000 // fps  # stable, monotonic
```

See [`utils/timestamps.py`](../utils/timestamps.py).

---

## Shooting Side Detection

When `SHOOTING_HAND = "auto"` in [`config/settings.py`](../config/settings.py):

- Compare left vs right wrist Y position in world space (higher Y = higher on body)
- The wrist that reaches higher during shooting is likely the shooting hand
- Result cached in `_resolved_side` for stability

Override with `SHOOTING_HAND = "left"` or `"right"` for known-handed players.

---

## Frame Buffer (Temporal Memory)

`FrameBuffer` stores the last 300 frames (~10 seconds at 30 FPS) as `FrameSnapshot` objects containing angles, shooting side, and phase.

**Current use:** Foundation for Phase 3 (phase detection).

**Future use:** Shot scoring, release timing analysis, movement trend detection.

---

## AI Concepts to Study

### Concept: Finite State Machine (FSM) — coming in Phase 3

**What it is:** A system with a fixed set of states and rules for transitioning between them based on inputs.

**Why we will use it:** A basketball shot has distinct phases (stance → loading → release → landing). An FSM models this naturally.

**Difficulty:** Beginner–Intermediate

**Topics to study:**
- State machines
- Hysteresis (preventing rapid state flicker)
- Signal derivative zero-crossing

---

### Concept: Ring Buffer

**What it is:** A fixed-size array where new items overwrite the oldest when full.

**Why we use it:** Stores recent frame history without unbounded memory growth.

**Difficulty:** Beginner

---

## Learning Roadmap

1. Trace `process_frame()` in [`pipeline.py`](../pipeline.py) line by line
2. Add a `print(frame_result)` in live mode to see per-frame output
3. Read `FrameBuffer` — understand how temporal data will feed phase detection
4. **Next phase:** Implement phase detection FSM using buffered angle derivatives

---

## Future Improvements

Phase 5 (shot scoring + aggregated feedback) is documented in [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md).
