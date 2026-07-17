# Phase 6 — Ball Tracking & Shot Outcome

> **Status:** Phase 6a ball/rim detection is integrated; release synchronization,
> make/miss classification, and reporting still require validation and completion.
> **Code:** [`ball/`](../ball/) (model loader, detector, tracker, timeseries,
> outcome experiments) + [`physics/`](../physics/)
> **Config:** [`config/ball.yaml`](../config/ball.yaml), [`config/hoop_roi.yaml`](../config/hoop_roi.yaml)
> **Depends on:** Phases 1–5b (pose, phases, rules, form scoring, session reports)
> **Study guide:** [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md)
> **Goal:** Detect the ball, track it through release and flight, and answer: *Did the player score?*

For CUDA, model selection, and the YOLO26/custom-weight distinction, read
[GPU_YOLO_SETUP.md](GPU_YOLO_SETUP.md) first.

---

## Why This Phase Exists

Swichy's pose branch answers **how** the player shot:

| Question | Current system | Phase 6 |
|----------|----------------|---------|
| Is form good? | Biomechanical rules + 0–100 form score | Same |
| When did release happen? | Phase FSM (`release` phase) | Refined with ball–wrist sync |
| Did the ball go in? | **Not measured** | **Make / miss / unknown** |
| Was form linked to outcome? | **Not measured** | Correlate form score vs result |

The current pipeline detects the ball and rim in every mode and carries those
results in `FrameResult`. What remains is a validated link from each completed
shot window to make/miss evidence:

1. **Form quality** — already built (Phases 4–5b)
2. **Performance outcome** — whether the attempt scored

This document separates the integrated detection/tracking work from the
remaining release, trajectory, outcome, and reporting work.

---

## Integrated Flow vs Remaining Flow

### Current integrated flow

```text
Camera
├─ BGR → basketball YOLO → ball + rim → ball tracker/time series
└─ RGB → pose → filter → angles → phases → rules → form score
                              ↓
                    FrameResult → HUD/recorder/report
```

Shot boundaries are inferred from **body phases**:

- Shot **starts:** `ready_stance → loading/ball_lift` OR **mid-entry** if recording starts mid-rep
- Shot **ends:** `landing → ready_stance` OR session `finalize_in_progress()`

`FrameResult` now includes raw ball/rim detections and a smoothed
`ball_snapshot`. The renderer draws these results, and the outcome classifier
receives a geometry-corrected rim region.

### Remaining Phase 6b–6d flow

```
Completed ShotSummary + ball time series
              ↓
      Release synchronization
              ↓
      Trajectory/outcome evidence
              ↓
      Outcome engine (make / miss / unknown)
              ↓
      Extended ShotSummary + SessionReport
```

The remaining fusion layer must link existing ball observations to the correct
shot window and preserve evidence when confidence is insufficient.

---

## What Phase 6 Must Deliver

### Functional requirements

1. **Ball/rim detection — integrated:** locate both objects in image space.
2. **Ball tracking — integrated:** smooth detections and bridge short gaps.
3. **Release sync — remaining:** align ball departure with body `release` (±N frames).
4. **Outcome classification — remaining:** `made` | `missed` | `unknown`.
5. **Timeseries evidence — partial:** buffer exists; report evidence is not fused.
6. **Mode parity — partial:** detections work in all modes; flight outcome requires
   a temporal source, so a single image cannot produce make/miss.

### Non-goals (v1)

- Exact 3D ball position without calibration
- Net physics simulation
- Multi-ball drills (one ball per shot window)
- Guaranteed hoop detection at arbitrary gym angles without target-camera validation

---

## High-Level Architecture

```mermaid
flowchart TD
    subgraph integrated [Integrated]
        A[Pose Detector] --> B[ShotAnalysisPipeline]
        B --> C[FrameResult]
        C --> D[ShotTracker]
        D --> E[ShotSummary — form only]
        F[Ball/Rim Detector] --> G[Ball Tracker]
        G --> H[BallTimeSeriesBuffer]
        H --> C
    end

    subgraph remaining [Remaining Phase 6b–6d]
        H --> I[ReleaseSync]
        I --> J[TrajectoryAnalyzer]
        J --> K[OutcomeClassifier]
    end

    C --> I
    E --> L[OutcomeFusion]
    K --> L
    L --> M[ShotOutcome]
    M --> N[Extended SessionReport]
```

### Phase 6 modules

| Module | Responsibility |
|--------|----------------|
| `ball/detector.py` | Multi-scale per-frame ball and rim detections |
| `ball/tracker.py` | Constant-velocity smoothing and short-gap prediction |
| `ball/timeseries.py` | Ring buffer of `BallSnapshot` (position, velocity, confidence) |
| `ball/release_sync.py` | Match ball departure to wrist landmarks + `release` phase |
| `ball/trajectory.py` | Fit parabolic arc; estimate apex, entry angle proxy |
| `ball/outcome.py` | Make/miss logic from timeseries + optional hoop ROI |
| `ball/models.py` | `BallSnapshot`, `BallTrajectory`, `ShotOutcome` dataclasses |
| `config/ball.yaml` | Detector thresholds, hoop ROI, outcome rules |
| `config/hoop_roi.yaml` | Optional manual hoop region per camera setup |

### Extensions to existing types

```python
# Implemented in ball/models.py (abridged)

@dataclass
class BallSnapshot:
    timestamp_ms: int
    frame_index: int
    center_xy: tuple[float, float]      # image pixels
    confidence: float
    velocity_xy: tuple[float, float]    # px/s from timeseries derivative
    state: str                         # in_hand | in_flight | at_rim | unknown

@dataclass
class ShotOutcome:
    result: str                        # made | missed | unknown
    confidence: float                  # 0–1
    release_frame: int | None
    release_timestamp_ms: int | None
    entry_frame: int | None            # ball crosses hoop plane proxy
    trajectory_apex_frame: int | None
    evidence: list[str]                # human-readable reasons
    timeseries_summary: dict           # compact stats for report

@dataclass
class ShotSummary:  # remaining extension in feedback/models.py
    # ... existing form fields ...
    outcome: ShotOutcome | None = None
```

`FrameResult` already exposes `ball`, `rim`, and `ball_snapshot`. The remaining
type change is attaching `ShotOutcome | None` to the completed `ShotSummary`.

---

## Time-Series: The Core of Outcome Detection

Outcome cannot be decided from one frame. Phase 6 treats the shot as a **sequence** of observations.

### Signal streams to record

| Stream | Source | Use |
|--------|--------|-----|
| Wrist Y (world) | Existing `KinematicFeatures` | Release timing candidate |
| Elbow extension rate | Existing angles derivative | Confirm release |
| Ball center (x, y) | Ball detector | Flight path |
| Ball velocity | `d(position)/dt` | Detect launch and rim crossing |
| Ball–wrist distance | Fused | `in_hand` vs `in_flight` transition |
| Hoop ROI occupancy | Ball position vs configured region | Make/miss at rim |

### Timeseries buffer design

Mirror the existing `FrameBuffer` pattern in [`utils/frame_buffer.py`](../utils/frame_buffer.py):

```
BallTimeSeriesBuffer
  - push(BallSnapshot) each frame
  - get_window(start_ms, end_ms) → list[BallSnapshot]
  - compute_velocity() → smoothed vx, vy (One Euro or Savitzky–Golay)
  - detect_events() → release, apex, rim_crossing
```

**Window alignment:** When `ShotTracker` closes a shot at `landing → ready_stance`, extend the analysis window by `post_shot_ms` (e.g. 1500 ms) to capture ball flight after the body lands.

### Event detection (rule-based v1)

| Event | Timeseries rule |
|-------|-----------------|
| **Release** | Ball–wrist distance exceeds threshold AND wrist velocity peak AND phase ≥ `release` |
| **In flight** | Ball confidence > τ for ≥ 3 consecutive frames after release |
| **Apex** | Ball Y velocity crosses zero (up → down) |
| **Rim approach** | Ball enters hoop ROI from above |
| **Made** | Ball center inside rim inner ellipse + downward velocity at crossing + no immediate exit |
| **Missed** | Ball crosses hoop plane but outside rim OR hits rim zone with high lateral velocity |
| **Unknown** | Low ball confidence, occlusion, or no hoop ROI configured |

Hysteresis and minimum-duration filters (same idea as phase FSM) prevent flicker.

---

## Ball Detection Options (Research → Production)

### Option A — Color + shape (fastest prototype)

- HSV orange segmentation + circularity filter
- **Pros:** No ML model, fast on CPU  
- **Cons:** Fails under mixed lighting, non-orange balls, motion blur

### Option B — Dedicated object detector (recommended v1)

- Fine-tuned YOLO / RT-DETR on basketball datasets  
- Classes: `ball`, optionally `rim`, `backboard`  
- **Pros:** Robust across backgrounds  
- **Cons:** Needs labeled data, GPU helps at high FPS

### Option C — MediaPipe / holistic + custom head

- Not available out of the box for ball; would still need custom detector

### Option D — Multi-camera triangulation

- See [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) #1 — late stage, highest accuracy

**Suggested path:** B for detection, rule-based timeseries for outcome in v1; optional ML outcome classifier in v2.

---

## Hoop / Rim Context

Make/miss requires a **goal reference**. Three tiers:

| Tier | Setup | Accuracy |
|------|-------|----------|
| **1. Manual ROI** | User draws hoop rectangle once per camera | Good for fixed backyard/hoop cam |
| **2. Rim detector** | YOLO class `rim` + auto ROI | Better for portable setups |
| **3. Calibrated court** | Homography to court plane | Best for gym / multi-angle |

v1 should support **Tier 1** (config file) so development is unblocked without full court vision.

Example future config:

```yaml
# config/hoop_roi.yaml (illustrative)
hoop:
  enabled: true
  roi_normalized: [0.42, 0.05, 0.58, 0.22]  # x1,y1,x2,y2 in 0–1
  rim_inner_scale: 0.72                       # shrink ROI for "made" zone
post_shot_capture_ms: 2000
```

---

## Fusion With Existing Shot Lifecycle

```mermaid
sequenceDiagram
    participant ST as ShotTracker
    participant BB as BallTimeSeriesBuffer
    participant RS as ReleaseSync
    participant OC as OutcomeClassifier
    participant SR as SessionRecorder

    ST->>ST: shot_in_progress = true
  loop Each frame
        BB->>BB: push(BallSnapshot)
    end
    ST->>ST: landing → ready_stance
    ST->>RS: shot window [t_start, t_end + post_shot]
    RS->>RS: find release event in window
    RS->>OC: trajectory + hoop ROI
    OC->>OC: made | missed | unknown
    OC->>SR: ShotOutcome attached to ShotSummary
    SR->>SR: REPORT.md includes outcome section
```

### Form score vs outcome score (conceptual)

Keep these **separate** so coaching stays honest:

| Metric | Meaning |
|--------|---------|
| **Form score** (existing) | Biomechanics quality 0–100 |
| **Outcome** (new) | Made or missed |
| **Performance index** (optional future) | Weighted combo for gamification |

A player can have **good form + miss** (rim variance) or **bad form + make** (lucky). Reports should show both.

---

## Report Extensions (Future)

Add to [`REPORT.md`](REPORTING.md) output:

```markdown
### Shot #1 — Form: 78/100 (Good) — Result: MADE ✓

- Release synced at 00:02.04 (frame 61)
- Ball apex at 00:02.31
- Entry angle (proxy): 52°
- Form–outcome note: Clean release; miss on previous shot was trajectory not elbow.
```

New key frame types:

- Release frame (ball leaving hand)
- Apex frame (highest ball position)
- Rim crossing frame (make or miss evidence)

---

## Implementation Phases (Suggested Order)

### 6a — Ball/rim detection prototype (integrated)

- [x] `ball/yolo_model.py` loads basketball-specific weights
- [x] `ball/detector.py` detects ball + rim with color fallback
- [x] Overlay ball/rim boxes in `visualization/renderer.py`
- [x] `BallTimeSeriesBuffer` stores tracked observations
- [x] Image/video/live modes pass frames to the ball branch
- [ ] Fine-tune `yolo26n.pt` on representative phone footage
- [ ] Establish labeled validation metrics across camera views

### 6b — Release synchronization

- [ ] `ball/release_sync.py` — fuse wrist distance + phase + ball velocity  
- [ ] Metrics: % shots where release frame aligns within ±2 frames of body release  
- [ ] Store `release_frame` on shot window

### 6c — Hoop ROI + rule-based outcome

- [ ] `config/hoop_roi.yaml` + UI or script to set ROI  
- [ ] `ball/outcome.py` — made / missed / unknown  
- [ ] Extend `ShotSummary` and session report  
- [ ] Unit tests with synthetic trajectories (parabolic curves through ROI)

### 6d — Trajectory analysis

- [ ] Parabolic fit, apex, entry angle proxy  
- [ ] Correlate entry angle with make/miss over a session (timeseries analytics)  
- [ ] Optional: link to [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) #11 physics model

### 6e — ML outcome classifier (optional v2)

- [ ] Train on labeled make/miss clips (trajectory features → label)  
- [ ] Fallback to rule-based when confidence low

---

## Testing Strategy (When Implemented)

| Test type | What it validates |
|-----------|-------------------|
| **Synthetic timeseries** | Parabolic `(x,y)` series → correct made/miss at rim |
| **Recorded clips** | Manually labeled `assets/` videos |
| **Sync test** | Release event within N frames of phase `release` |
| **Edge cases** | Occluded ball, air ball, rim bounce, no hoop in frame |

No implementation in this phase — tests are listed for future work.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Motion blur hides ball | Unknown outcome | Interpolate between high-confidence frames; widen `unknown` band |
| Fixed camera not showing hoop | Cannot score outcome | Require hoop in frame or disable outcome for that session |
| Orange clutter (shirts, signs) | False ball detections | Temporal consistency filter; size prior; ML detector |
| Monocular depth | Weak 3D arc | Use image-plane trajectory + ROI; defer 3D to multi-cam |
| Latency (two models) | Drops FPS | Run ball detector every 2nd frame; Kalman predict between |

---

## Dependencies & Prerequisites

Before starting Phase 6:

- [x] Stable shot window from `ShotTracker`  
- [x] `FrameBuffer` / per-shot frame history  
- [x] Session reports (`SessionRecorder`)  
- [x] CUDA-capable Ultralytics runtime and custom basketball weights
- [x] Ball/rim detection, tracking, overlays, and verification scripts
- [ ] Labeled ball positions on sample videos (even 50 frames manual)  
- [x] Hoop visible in test videos
- [x] Detector selected: custom E-BARD YOLO with multi-scale inference

**Python packages:** `ultralytics`, CUDA-enabled PyTorch, OpenCV, and NumPy are
installed and verified. See [GPU_YOLO_SETUP.md](GPU_YOLO_SETUP.md).

---

## Related Documents

| Doc | Relation |
|-----|----------|
| [PIPELINE.md](PIPELINE.md) | Where ball branch attaches after pose |
| [PHASE_DETECTION.md](PHASE_DETECTION.md) | Release phase used for sync |
| [REPORTING.md](REPORTING.md) | Reports extended with outcome section |
| [FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md) | #6 Ball-body sync, #11 trajectory physics |
| [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md) | Roadmap slot for Phase 6 |

---

## Summary

Phase 6 closes the loop between **shooting form** and **shooting results**.
Phase 6a already provides a parallel ball/rim pipeline, tracking, overlays, and
time-series observations. The next work is Phase 6b–6d: validate release sync,
classify make/miss with explicit evidence, attach the result to `ShotSummary`,
and include it in session reports.
