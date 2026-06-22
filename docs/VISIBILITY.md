# Visibility and Occlusion Handling

## What Changed

| File | Change |
|------|--------|
| [`pose/visibility.py`](../pose/visibility.py) | **New** — `VisibilityGate` class with confidence thresholding and temporal hold |
| [`angles/calculator.py`](../angles/calculator.py) | Returns `is_valid=False` when any landmark in a chain is unreliable |
| [`visualization/renderer.py`](../visualization/renderer.py) | Shows "N/A" for invalid angles, `~` prefix for unstable (held) angles |

---

## Why It Changed

MediaPipe assigns a **visibility** score (0–1) to each landmark indicating how likely it is visible in the frame. When a landmark is occluded (arm behind body, player turned away, another person blocking), the position estimate becomes unreliable.

Computing angles from bad positions produces **confidently wrong numbers** — worse than showing nothing. A knee angle of 45° from a guessed occluded ankle could trigger false coaching.

---

## Strategy

### 1. Per-landmark confidence gate

```python
reliable = visibility >= threshold  # default 0.6
```

Configurable via `VISIBILITY_THRESHOLD` in [`config/settings.py`](../config/settings.py).

### 2. Angle chain validation

A joint angle requires three landmarks (proximal, vertex, distal). If **any** is unreliable → entire angle is invalid (`degrees=None`, `is_valid=False`).

### 3. Temporal hold

Brief occlusion (1–5 frames) is common during fast motion. Instead of immediately invalidating:

- Hold the last reliable position
- Decay confidence linearly: `visibility × (1 - hold_count / (hold_frames + 1))`
- Mark `is_stable=False` so UI shows approximate value with `~` prefix

If occlusion persists beyond `VISIBILITY_HOLD_FRAMES` (default 5), stop holding and mark unreliable.

### 4. UI feedback

| State | Display | Color |
|-------|---------|-------|
| Valid + stable | `Elbow: 145` | Green |
| Valid + unstable (held) | `Elbow: ~145` | Orange |
| Invalid | `Elbow: N/A` | Gray |

---

## Camera View Considerations

| View | Typical issues | Mitigation |
|------|---------------|------------|
| Side view | Far arm occluded by torso | Use visible side; auto-detect shooting arm |
| Back view | Face/wrist less visible | Trunk and leg angles still work; arm angles may be N/A |
| Diagonal | Partial occlusion | Visibility gate + hold |

---

## AI Concepts to Study

### Concept: Confidence-Aware Inference

**What it is:** Using model-provided confidence scores to decide whether to trust a prediction.

**Why we use it:** Neural networks always output a number — confidence tells you when that number is a guess.

**Alternatives:** Hard threshold only, Bayesian uncertainty, ensemble disagreement.

**Difficulty:** Beginner

**Topics to study:**
- Softmax confidence vs calibration
- Missing data handling
- Sensor fusion with uncertainty

---

## Learning Roadmap

1. Run live mode, turn away from camera — observe angles become N/A
2. Wave arm quickly — observe brief `~` unstable values during fast motion
3. Tune `VISIBILITY_THRESHOLD` in settings — lower = more angles shown but less reliable

---

## Pipeline Position

```
One Euro Filter
        ↓
Visibility Gate  ← YOU ARE HERE
        ↓
3D Angle Computation
```
