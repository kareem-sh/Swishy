# 3D Angle Computation

## What Changed

| File | Change |
|------|--------|
| [`geometry/vectors.py`](../geometry/vectors.py) | **New** — 3D vector math: `angle_between_vectors`, `segment_vector`, `normalize`, `midpoint`, `angle_from_vertical` |
| [`angles/calculator.py`](../angles/calculator.py) | **New** — `AngleCalculator` computes joint angles from world landmarks |
| [`angles/joint_chains.py`](../angles/joint_chains.py) | **New** — Named joint definitions (elbow, knee, hip, shoulder, trunk) |
| [`pose/landmarks.py`](../pose/landmarks.py) | **New** — `extract_world_landmarks()` reads `pose_world_landmarks` from MediaPipe |
| [`core/angles.py`](../core/angles.py) | **Deprecated** — old 2D `arctan2` method kept for reference only |

---

## Why It Changed

### The 2D problem

The old code computed angles like this:

```python
radians = arctan2(c_y - b_y, c_x - b_x) - arctan2(a_y - b_y, a_x - b_x)
```

This measures the angle **in the image plane** (the 2D projection of the body onto the camera sensor). When the camera angle changes, the projection changes — even if the real joint angle stays the same.

**Example:** A fully extended elbow (180°) filmed from the side looks like ~150° in 2D if the arm points slightly toward the camera. Filmed from the front, the same arm might look like 170°. The real angle did not change — the projection did.

### The 3D solution

We now use **MediaPipe world landmarks** — 3D coordinates in meters, centered on the hip midpoint, in a body-centric frame:

- **Origin:** Midpoint between hips
- **+Y:** Up (toward head)
- **+X:** Subject's right
- **+Z:** Toward the camera

Angles are computed with **3D vector dot product**, which is rotation-invariant:

```
v1 = A - B    (vector from vertex to proximal landmark)
v2 = C - B    (vector from vertex to distal landmark)

cos(θ) = (v1 · v2) / (|v1| × |v2|)
θ = arccos(cos(θ))
```

This angle does **not change** when you rotate the coordinate frame — only when the actual joint flexes.

---

## Coordinate Systems Compared

| System | Source | Units | Camera-independent? | Used for |
|--------|--------|-------|--------------------|---------|
| Image `(x, y)` | `pose_landmarks` | Normalized 0–1 | No | Drawing skeleton on screen |
| Image `(x, y, z)` | `pose_landmarks` | Mixed | Partially | Not used |
| **World `(x, y, z)`** | `pose_world_landmarks` | Meters | **Best with single camera** | **All angle computation** |
| Calibrated multi-cam 3D | External calibration | Meters | Yes (near ground truth) | Future improvement |

### Honest limitation

World landmarks are estimated from a **single monocular camera**. They are much better than 2D but not biomechanically perfect. For a graduation project and early commercial prototype, they are the right tradeoff.

---

## Joint Chains

Each angle is defined by three landmarks where the **middle landmark is the vertex**:

| Angle Name | Proximal (A) | Vertex (B) | Distal (C) |
|------------|-------------|------------|------------|
| `right_elbow` | right_shoulder | right_elbow | right_wrist |
| `right_knee` | right_hip | right_knee | right_ankle |
| `right_hip` | right_shoulder | right_hip | right_knee |
| `right_shoulder` | right_hip | right_shoulder | right_elbow |
| `right_index_align` | right_elbow | right_wrist | right_index |
| `trunk` | mid_hip | mid_shoulder | vertical reference |

Left-side chains mirror right-side. The shooting side (auto-detected or configured) determines which side is analyzed.

**Index alignment** measures how straight the shooting finger is relative to the forearm — used for release and follow-through rules and phase detection.

---

## AngleResult

Every computed angle returns:

```python
@dataclass
class AngleResult:
    name: str              # e.g. "right_elbow"
    degrees: float | None  # None if landmarks unreliable
    is_valid: bool           # False if any landmark occluded
    is_stable: bool          # False if using held (decayed) position
```

- **`is_valid = False`** → UI shows "N/A" instead of a wrong number
- **`is_stable = False`** → UI shows `~145` (approximate) in orange

---

## AI Concepts to Study

### Concept: 3D Vector Dot Product for Angles

**What it is:** The dot product of two vectors equals the product of their magnitudes times the cosine of the angle between them: `a · b = |a| |b| cos(θ)`.

**Why we use it:** Computing `θ = arccos((a · b) / (|a| |b|))` gives the true 3D angle regardless of camera orientation.

**Alternatives:**
- 2D `arctan2` (old approach — camera-dependent)
- Cross product magnitude: `θ = arcsin(|a × b| / (|a| |b|))` (equivalent, less numerically stable near 0°/180°)
- Quaternion-based rotation (overkill for joint angles)

**Advantages:** Rotation-invariant, mathematically exact in 3D, simple to implement.

**Disadvantages:** Requires reliable 3D coordinates (not just 2D image points).

**Mathematical intuition:** Imagine two sticks joined at a hinge. The angle between them is a property of the sticks in space — it does not depend on where you stand to look at them.

**Difficulty:** Beginner (linear algebra)

**Topics to study:**
- Dot product and cross product
- Vector normalization
- Rotation invariance
- Interior vs exterior angles

**Resources to search:**
- "3D vector angle dot product"
- "arccos clip numerical stability"
- "joint angle biomechanics 3D"

---

### Concept: MediaPipe World Landmarks

**What it is:** MediaPipe estimates 3D body joint positions in a canonical body-centric coordinate frame, derived from a single RGB image.

**Why we use it:** Provides 3D coordinates without depth cameras or multi-camera calibration.

**Alternatives:**
- Image landmarks only (2D — rejected)
- Depth camera (Intel RealSense, LiDAR)
- Multi-camera triangulation
- SMPL body model fitting

**Advantages:** Works with any webcam, real-time, no extra hardware.

**Disadvantages:** Monocular depth estimation is approximate; fast motion can cause drift.

**Difficulty:** Intermediate

**Topics to study:**
- Monocular 3D pose estimation
- MediaPipe Pose Landmarker API
- `pose_landmarks` vs `pose_world_landmarks`

**Resources to search:**
- "MediaPipe pose world landmarks"
- "monocular 3D human pose estimation survey"

---

## Learning Roadmap

1. **Linear algebra refresh:** dot product, vector magnitude, normalization (Khan Academy / 3Blue1Brown)
2. **Implement by hand:** compute angle between two 3D vectors on paper, verify with `tests/test_angles.py`
3. **Compare 2D vs 3D:** run old `core/angles.calculate_angle` and new `AngleCalculator` on the same frame — observe the difference at different camera angles
4. **Read:** MediaPipe Pose Landmarker documentation for world landmark coordinate system
5. **Next:** Study `FILTERS.md` — smoothing before angle computation reduces noise amplification

---

## Pipeline Position

```
World landmarks extracted
        ↓
One Euro Filter (smooth positions)
        ↓
Visibility Gate (reject unreliable)
        ↓
3D Angle Computation
        ↓
Phase Detection + Rules
        ↓
Reports (visibility gap notes if occluded too long)
```
