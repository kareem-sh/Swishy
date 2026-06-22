# Landmark Filtering — One Euro Filter

## What Changed

| File | Change |
|------|--------|
| [`filters/one_euro.py`](../filters/one_euro.py) | **New** — `OneEuroFilter` (single axis) and `LandmarkFilterBank` (all landmarks) |
| [`config/filter_config.yaml`](../config/filter_config.yaml) | **New** — Tunable filter parameters |
| [`pipeline.py`](../pipeline.py) | Filter applied to world landmarks before visibility gating and angle computation |

---

## Why It Changed

### The jitter problem

MediaPipe outputs a new landmark position every frame (~30 FPS). Even when you stand still, coordinates fluctuate by small amounts due to:

- Neural network quantization noise
- Image compression artifacts
- Lighting changes
- Auto-exposure shifts

When you compute angles from noisy positions, the noise **amplifies** — a 1-pixel wrist jitter can cause a 5–10° elbow angle swing. This makes coaching feedback unreliable ("Elbow Too Bent" flickering on and off).

### The solution: temporal smoothing

We apply a **low-pass filter** to each landmark coordinate (x, y, z) before computing angles. The filter removes high-frequency noise while preserving real movement.

---

## Filter Evaluation

We evaluated four filtering strategies for real-time basketball:

| Filter | Complexity | Latency | Fast Motion | Verdict |
|--------|-----------|---------|-------------|---------|
| **One Euro** | O(1) per value | ~1 frame | Adaptive — responsive during fast motion | **Selected** |
| Kalman | O(1), needs state model | Low–medium | Good if Q/R tuned | Rejected — tuning burden |
| EMA | O(1) | Fixed lag | Sluggish during release | Rejected — too slow for jump shots |
| Savitzky-Golay | O(window) | window/2 frames | Smooth but delayed | Rejected — kills release timing |

### Why One Euro was selected

Basketball jump shots have **two types of motion**:

1. **Slow phases** (ready stance, loading) — want heavy smoothing
2. **Fast phases** (release, follow-through) — want low latency

The One Euro Filter adapts its cutoff frequency based on **signal speed**:

- When the landmark is still → low cutoff → heavy smoothing
- When the landmark moves fast → high cutoff → follows motion closely

This is exactly what basketball needs.

### Why Kalman was rejected

Kalman filters are excellent when you have a good **motion model** (e.g. constant velocity, constant acceleration). Human basketball motion is non-linear and phase-dependent. Tuning the process noise (Q) and measurement noise (R) matrices for 13 landmarks × 3 axes would require extensive calibration with little gain over One Euro.

### Why EMA was rejected

Exponential Moving Average uses a fixed smoothing factor α. A high α (responsive) lets noise through. A low α (smooth) adds lag that blurs the release moment — the most critical phase for coaching.

### Why Savitzky-Golay was rejected

Savitzky-Golay fits a polynomial over a sliding window of frames. It produces excellent smooth curves but requires **future frames** (or adds window_size/2 frames of latency). For offline video analysis it would be great; for real-time coaching it delays release detection unacceptably.

---

## How One Euro Works

### Mathematical intuition

A low-pass filter removes high-frequency components (jitter) from a signal. The key question is: **how aggressively to filter?**

One Euro answers this adaptively:

```
cutoff = min_cutoff + beta × |speed|
```

- `min_cutoff` — minimum cutoff frequency (Hz). Lower = more smoothing when still.
- `beta` — speed coefficient. Higher = less lag during fast motion.
- `|speed|` — estimated derivative (how fast the signal is changing).

When you are in ready stance (speed ≈ 0): cutoff ≈ min_cutoff → heavy smoothing.
When you release the ball (speed high): cutoff increases → filter follows your wrist.

### Implementation

Each landmark has 3 independent One Euro filters (x, y, z). At 30 FPS with 13 basketball landmarks, that is 39 filter instances — all O(1) per frame.

Default parameters in [`config/filter_config.yaml`](../config/filter_config.yaml):

```yaml
min_cutoff: 1.0    # Baseline smoothing
beta: 0.007        # Speed responsiveness
d_cutoff: 1.0      # Derivative smoothing
```

Tune `beta` upward if the filter feels sluggish during release. Tune `min_cutoff` downward if angles jitter during stance.

---

## AI Concepts to Study

### Concept: One Euro Filter

**What it is:** An adaptive low-pass filter that adjusts its cutoff frequency based on the speed of the input signal. Published by Casiez, Roussel, and Vogel (2012).

**Why we use it:** Optimal balance of smoothness and responsiveness for interactive real-time systems.

**Alternatives:** Kalman, EMA, Savitzky-Golay, Butterworth, median filter.

**Advantages:** O(1) complexity, no training, adapts to motion speed, widely used in pose estimation.

**Disadvantages:** Requires parameter tuning per application; not optimal for offline batch processing.

**Mathematical intuition:** Imagine a spring connecting the filtered value to the raw value. When you move slowly, the spring is stiff (stays smooth). When you move fast, the spring loosens (follows your motion).

**Difficulty:** Intermediate

**Topics to study:**
- Low-pass filters
- Cutoff frequency
- Adaptive filtering
- Signal processing fundamentals
- Frequency domain (Fourier transform)

**Resources to search:**
- "1€ Filter paper Casiez"
- "adaptive low pass filter real time"
- "One Euro Filter pose estimation"

---

### Concept: Low-Pass Filter

**What it is:** A filter that attenuates high-frequency components of a signal, keeping low-frequency (slow) changes.

**Why we use it:** Landmark jitter is high-frequency noise; real body movement during a shot is lower frequency.

**Alternatives:** High-pass (keeps fast changes), band-pass (keeps a range), notch (removes specific frequency).

**Advantages:** Simple concept, effective for noise reduction.

**Disadvantages:** Always introduces some lag; aggressive filtering blurs fast movements.

**Difficulty:** Beginner

**Topics to study:**
- Time domain vs frequency domain
- Convolution
- Moving average as simplest low-pass

---

## Computational Cost

| Operation | Per frame | At 30 FPS |
|-----------|----------|-----------|
| One Euro per axis | ~10 floating-point ops | trivial |
| 39 filters (13 landmarks × 3) | ~390 ops | < 0.1 ms on modern CPU |

Filtering is not a performance bottleneck. Pose detection (MediaPipe neural network) dominates runtime.

---

## Learning Roadmap

1. **Read the paper:** "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems" (Casiez et al., 2012)
2. **Experiment:** In `config/filter_config.yaml`, set `beta: 0.0` — observe that the filter becomes a fixed low-pass (like EMA). Then set `beta: 0.05` — observe faster response during motion.
3. **Visual test:** Run live mode, hold still (angles should be stable), then move fast (angles should follow without large lag).
4. **Next:** Study visibility gating in `pose/visibility.py` — filtering cannot fix occluded landmarks.

---

## Pipeline Position

```
World landmarks extracted
        ↓
One Euro Filter  ← YOU ARE HERE
        ↓
Visibility Gate
        ↓
3D Angle Computation
        ↓
Frame Buffer
```
