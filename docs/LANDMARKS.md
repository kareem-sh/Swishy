# Pose Landmarks: Hands, Feet, and Reliability

This guide explains which MediaPipe Pose landmarks Swichy uses, how they move
through the pipeline, and which measurements are safe enough to affect coaching.

## Image space and world space

MediaPipe returns two representations:

- **Image landmarks** use normalized image coordinates. Swichy converts them to
  pixels for the skeleton and overlays.
- **World landmarks** use a body-centered 3D coordinate system. Swichy uses
  these points for angles, velocities, phase detection, and biomechanical rules.

The same landmark must also pass visibility and presence checks. If one point in
an angle is unreliable, the angle is marked invalid instead of becoming a
coaching failure.

## Index-finger flow

MediaPipe Pose provides one index-tip point per hand. It does not provide every
finger joint like MediaPipe Hands.

```text
right_index (tip)
  → landmark extraction
  → One Euro smoothing
  → visibility/presence gate
  → elbow–wrist–index alignment angle
  → index vertical velocity
  → release/follow-through phase evidence
  → release and follow-through coaching rules
  → score, HUD warning, and report
```

The angle definition is:

```python
JointChain(
    proximal="right_elbow",
    vertex="right_wrist",
    distal="right_index",
)
```

A nearly straight elbow–wrist–index line approaches 180 degrees. Swichy uses
this as supporting evidence for wrist/finger extension at release. It is not a
measurement of individual finger-joint curl.

If the index tip is hidden, Swichy now sets index velocity to zero and invalidates
the alignment angle. It does not copy wrist motion into the finger signal.

## Foot and shoe-tip flow

MediaPipe Pose exposes `heel` and `foot_index` for each side. `foot_index`
represents the visible front of the foot or shoe, so basketball shoes do not
prevent the landmark from being useful.

Swichy extracts and filters these points and computes:

```python
JointChain(
    proximal="right_knee",
    vertex="right_ankle",
    distal="right_foot_index",
)
```

This is a coaching-grade ankle-flexion proxy, not a medical measurement. It is
available as `right_ankle_flexion` or `left_ankle_flexion` in `FrameResult.angles`.

Example:

```python
ankle = frame_result.angles["right_ankle_flexion"]
if ankle.is_valid:
    print(f"Ankle flexion: {ankle.degrees:.1f}°")
else:
    print("Ankle flexion unavailable: foot landmark is occluded")
```

## Why foot angle does not affect the score yet

Adding the angle improves observability, but immediately scoring it would reduce
reliability because:

1. the shoe tip is frequently outside the frame or hidden by the other leg;
2. front and diagonal camera views distort a 2D-looking foot orientation;
3. no project dataset currently defines validated ankle-flexion ranges by phase;
4. footwear changes the visible shoe outline, even though the landmark still
   tracks the front of the foot.

Therefore:

- Jump and landing detection continue to use ankle height and velocity.
- Ankle flexion is visibility-gated diagnostic data.
- It should become a coaching rule only after labeled side-view footage shows
  that valid measurements are repeatable and useful.

## Validation protocol before scoring foot angle

Record at least 30 side-view attempts on target phones:

1. Include multiple players and shoe styles.
2. Keep the full feet visible.
3. Record loading, takeoff, release, and landing.
4. Count valid ankle-flexion frames per phase.
5. Compare repeated attempts from the same player.
6. Add a rule only if the signal is stable and predicts a coaching-relevant
   mistake better than ankle height alone.

Suggested acceptance criteria:

- at least 80% valid measurements in loading and landing;
- less than 10 degrees median variation across repeated similar attempts;
- no phase or score regression when the foot leaves the frame.

## Relevant code

- `pose/landmarks.py` — active landmark subset.
- `pose/visibility.py` — reliability gating.
- `angles/joint_chains.py` — index and ankle chains.
- `angles/calculator.py` — visibility-gated 3D angle calculation.
- `phase_detection/features.py` — wrist, index, ankle, and body velocities.
- `phase_detection/detector.py` — index-assisted release transitions.
- `config/biomechanics.yaml` — index coaching rules.
- `tests/test_angles.py` and `tests/test_phases.py` — executable examples.
