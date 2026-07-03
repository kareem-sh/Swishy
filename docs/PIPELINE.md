# Analysis Pipeline

Central orchestrator: [`pipeline.py`](../pipeline.py) → `ShotAnalysisPipeline.process_frame()`.

All modes (`live_stream`, `video_mode`, `image_mode`) call the same pipeline and produce identical `FrameResult` objects.

---

## Data Flow

```
Input (camera / video / image)
    ↓
pose/detector.py          MediaPipe Pose Landmarker
    ↓
pose/landmarks.py         image + world landmark dicts
    ↓
filters/one_euro.py       smoothed world coordinates
    ↓
pose/visibility.py        reliable flags + temporal hold
    ↓
angles/calculator.py      Dict[str, AngleResult] incl. index_align
    ↓
phase_detection/features.py   KinematicFeatures (velocities, index)
    ↓
phase_detection/detector.py   phase string (8 states)
    ↓
analysis/engine.py        AnalysisResult (rules for this phase)
    ↓
feedback/shot_tracker.py  collect frames; ShotSummary on complete
    ↓
visualization/hud_display.py  smooth phase/angles for overlay
    ↓
FrameResult
    ↓
visualization/renderer.py + hud.py
    ↓
feedback/session_recorder.py  (video/live/image) → PDF report
```

---

## `FrameResult` Fields

```python
@dataclass
class FrameResult:
    image_landmarks: Optional[dict]
    world_landmarks: Optional[dict]
    angles: Dict[str, AngleResult]
    features: Optional[KinematicFeatures]
    analysis: Optional[AnalysisResult]
    shooting_side: str                    # "left" | "right"
    phase: str                          # e.g. "release"
    phase_label: str                      # e.g. "Release"
    timestamp_ms: int
    has_pose: bool
    shot_in_progress: bool
    last_shot_score: Optional[int]
    shot_summary: Optional[ShotSummary]   # set when shot just completed
    display_summary: Optional[ShotSummary]
    show_shot_summary: bool
    hud_display: Optional[HudDisplay]     # smoothed values for readable HUD
    capture_warning: Optional[str]        # mid-entry warning during shot
```

---

## Usage

```python
from pipeline import ShotAnalysisPipeline

pipeline = ShotAnalysisPipeline(shooting_hand="auto")
pipeline.set_fps(30.0)

result = pipeline.process_frame(detection_result, width, height, timestamp_ms)

if result.has_pose and result.angles["right_elbow"].is_valid:
    print(result.phase_label, result.angles["right_elbow"].degrees)

if result.shot_summary:
  print(f"Shot done: {result.shot_summary.score}/100")
```

---

## Timestamps

MediaPipe VIDEO/LIVE_STREAM modes need **monotonic** timestamps:

```python
# utils/timestamps.py
timestamp_ms = int(frame_index * 1000.0 / fps)
```

Do not use `time.time()` for video files.

---

## Shooting Side

`SHOOTING_HAND` in [`config/settings.py`](../config/settings.py):
- `"auto"` — higher wrist in world Y wins
- `"left"` / `"right"` — force side for consistent elbow/knee/index rules

---

## Session End

```python
pipeline.finalize_session()  # scores in-progress shot if video ended mid-rep
```

Called from `SessionRecorder.finalize()` when video ends before `landing → ready_stance`.

---

## Frame Buffer

`utils/frame_buffer.py` stores last ~300 `FrameSnapshot` objects (angles, phase, analysis) for temporal use and shot scoring.

---

## HUD Display Smoothing

Raw per-frame values change too fast on video. `visualization/hud_display.py` applies:
- Phase label hold (stable frames before switching text)
- Angle EMA + minimum step
- Violation message persistence

Tuned in [`config/display.yaml`](../config/display.yaml).

---

## Related

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md)
- [REPORTING.md](REPORTING.md)
