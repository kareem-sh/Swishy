# Analysis Pipeline

[`pipeline.py`](../pipeline.py) is the application orchestrator. Video, live,
and image modes all call `ShotAnalysisPipeline.process_frame()` and receive the
same `FrameResult` contract.

## Why the pipeline exists

The pipeline owns coordination and temporal state. Individual modules own one
kind of logic:

- pose modules detect and normalize body landmarks,
- ball modules detect and track court objects,
- phase modules understand motion order,
- analysis modules evaluate biomechanics,
- feedback modules aggregate a shot,
- visualization modules draw results.

This separation keeps model loading, pose math, coaching rules, and rendering
from becoming one large function.

## Input contract

```python
frame_result = pipeline.process_frame(
    detection_result=pose_result,
    width=frame_width,
    height=frame_height,
    timestamp_ms=timestamp_ms,
    bgr_frame=frame,          # enables ball/rim YOLO
)
```

- `detection_result` is the MediaPipe result.
- `width` and `height` convert normalized image landmarks to pixels.
- `timestamp_ms` must increase monotonically.
- `bgr_frame` is optional so pose-only consumers remain supported.

## Parallel frame flow

```text
Input frame
├─ BGR branch
│  └─ ball/detector.py
│     └─ custom basketball YOLO
│        ├─ BallDetection → BallTracker → BallSnapshot → BallTimeSeriesBuffer
│        └─ RimDetection → live hoop region for outcome logic
│
└─ RGB / pose branch
   └─ pose/detector.py
      └─ pose/landmarks.py
         └─ filters/one_euro.py
            └─ pose/visibility.py
               └─ angles/calculator.py
                  └─ phase_detection/features.py
                     └─ phase_detection/detector.py
                        └─ analysis/engine.py
                           └─ feedback/shot_tracker.py

Branches merge → FrameResult → renderer/HUD → session recorder/report
```

Ball/rim detection still produces a result when pose is absent. This matters
when the player leaves the frame but the ball continues toward the hoop.

## Processing order

### 1. Ball and rim

When `config/ball.yaml` enables detection and a BGR frame is provided:

1. `BallDetector` runs the custom E-BARD basketball model.
2. `device: auto` selects CUDA GPU 0 or CPU fallback.
3. `frame_stride` skips expensive inference frames.
4. The previous detection is held between inference frames with reduced
   confidence.
5. The strongest rim is retained for a static camera (`sticky_rim`).
6. `BallTracker` smooths the center and fills short detection gaps.
7. `BallTimeSeriesBuffer` stores snapshots for trajectory/outcome work.

### 2. Pose landmarks

`extract_all_landmarks()` returns image-space pixels and world-space metres.
If there is no pose, the pipeline returns early while preserving ball/rim data.

### 3. Filtering and reliability

The One Euro filter smooths world coordinates. `VisibilityGate` marks
low-confidence joints unreliable and can hold a recent reliable value briefly.

### 4. Angles and features

`AngleCalculator` produces stable 3D angles. `extract_features()` combines
current and previous frames into velocities, angle changes, and stillness.

### 5. Phase FSM

`ShotPhaseDetector` evaluates motion thresholds, valid transition order,
hysteresis, and minimum dwell. See
[PHASE_DETECTION.md](PHASE_DETECTION.md).

### 6. Biomechanics and shot aggregation

`BiomechanicsEngine` evaluates only rules relevant to the current phase.
`ShotTracker` collects results and creates a `ShotSummary` when a shot closes.

### 7. Display and recording

`HudDisplaySmoother` makes fast-changing values readable. The renderer draws
pose, ball, rim, and HUD. `SessionRecorder` captures frames and reports.

## `FrameResult`

Important fields:

```python
@dataclass
class FrameResult:
    image_landmarks: Optional[dict]
    world_landmarks: Optional[dict]
    angles: Dict[str, AngleResult]
    features: Optional[KinematicFeatures]
    analysis: Optional[AnalysisResult]
    phase: str
    phase_label: str
    timestamp_ms: int
    has_pose: bool
    shot_in_progress: bool
    shot_summary: Optional[ShotSummary]
    hud_display: Optional[HudDisplay]
    ball: Optional[BallDetection]
    rim: Optional[RimDetection]
    ball_snapshot: Optional[BallSnapshot]
```

Consumers should use these public fields instead of accessing a detector or
tracker's private attributes.

## Temporal state

The pipeline owns:

- previous timestamp and world landmarks,
- filtered landmark state,
- visibility hold state,
- current phase and transition counters,
- ankle standing baseline,
- shot aggregation state,
- body frame buffer,
- ball detector/tracker/timeseries state,
- sticky rim state,
- HUD smoothing state.

Call `pipeline.reset()` when changing session or camera. This clears body and
ball state together.

## Timestamps

MediaPipe video/live modes require monotonic timestamps:

```python
timestamp_ms = int(frame_index * 1000.0 / fps)
```

Do not use wall-clock time for a video file. Feature velocities depend on
`dt`, so invalid timestamps also damage phase detection.

## Configuration

| File | Controls |
|------|----------|
| `config/filter_config.yaml` | landmark smoothing |
| `config/phases.yaml` | FSM thresholds and temporal confirmation |
| `config/biomechanics.yaml` | phase-specific form rules |
| `config/scoring.yaml` | scoring behavior |
| `config/display.yaml` | HUD/overlay timing and visibility |
| `config/ball.yaml` | model, device, thresholds, image size, frame stride |
| `config/hoop_roi.yaml` | optional manual hoop region |

## Current Phase 6 boundary

Integrated now:

- custom ball + hoop inference,
- automatic GPU selection,
- ball tracking and short-gap interpolation,
- ball timeseries storage,
- rim/ball overlays in image, video, and live modes,
- live rim geometry supplied to outcome logic.

Still incomplete:

- robust release synchronization with pose,
- validated make/miss classification,
- attaching `ShotOutcome` to the production `ShotSummary` and reports,
- phone-camera fine-tuning and ONNX export.

Keeping this boundary explicit prevents an experimental outcome rule from being
presented as a reliable product result.

## Verification

Pose/phase:

```powershell
python tests/test_angles.py
python tests/test_phases.py
python tests/test_feedback.py
```

Ball/rim:

```powershell
python scripts/verify_ball_rim.py --source assets/images/image_03_basketball_shoot.jpg --max-frames 1
python scripts/verify_ball_rim.py --source assets/videos/video_03_expert_score.mp4 --max-frames 120
python scripts/verify_pipeline.py --max-frames 120
```

Environment and CUDA setup:
[GPU_YOLO_SETUP.md](GPU_YOLO_SETUP.md).
