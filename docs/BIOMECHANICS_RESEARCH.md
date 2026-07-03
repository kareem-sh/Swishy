# Basketball Shooting Biomechanics — Research References

This document summarizes peer-reviewed literature that informs Swichy's phase detection thresholds, joint-angle rules, and coaching messages. Rules in [`config/biomechanics.yaml`](../config/biomechanics.yaml) and phase thresholds in [`config/phases.yaml`](../config/phases.yaml) are tuned against these findings.

---

## Why This Matters for Swichy

Swichy measures 3D joint angles from pose landmarks and evaluates form **by shot phase**. Research consistently shows that shooting success depends on:

1. **How much** key joints flex/extend (magnitude)
2. **When** peak motions occur (timing / kinematic sequence)
3. **How stable** the trunk and head stay through release

Our rule engine encodes (1) as phase-gated angle ranges. Our FSM encodes (2) as wrist, elbow, index-finger, and ankle kinematics. Reports flag (3) via visibility-gap notes when landmarks are occluded.

---

## Core Papers

### 1. Made vs Missed Jump Shots (Recreational Male Players)

**Citation:** Jovanović, M., et al. (2022). *Differences in Biomechanical Characteristics between Made and Missed Jump Shots in Male Basketball Players.* Biomechanics, 2(3), 428–441.

**DOI:** [10.3390/biomechanics2030028](https://doi.org/10.3390/biomechanics2030028)

**Key findings applied in Swichy:**

| Finding | Swichy implementation |
|---------|-------------------------|
| Higher elbow position in preparatory phase → more makes (2PT) | `elbow_slot_ball_lift` rule (75–115°), coaching on elbow-under-ball |
| More erect torso on made 3PT shots | `trunk_posture` max lean tightened to 22° |
| Greater release angle & release height on made 3PT | `release_height` min raised; release phase uses elbow + index extension |
| Greater knee/hip flexion before release at longer range | `knee_flexion_loading` range 70–125° |

---

### 2. Distance & Proficiency Effects (Professional Players)

**Citation:** Jovanović, M., et al. (2023). *Impact of Distance and Proficiency on Shooting Kinematics in Professional Male Basketball Players.* Sports, 7(4), 78.

**URL:** [MDPI Sports 7(4):78](https://www.mdpi.com/2411-5142/7/4/78)

**Key findings applied in Swichy:**

| Finding | Swichy implementation |
|---------|-------------------------|
| Three-point shots use more knee/hip flexion than free throws | Loading-phase knee/hip rules fire during `loading` and `knee_flexion` |
| Release height greater on jump shots vs set shots | `release_height` metric (wrist relative to head) |
| Release angle lower at 3PT distance | Documented for future distance-aware rule profiles |
| Sagittal (side) camera view used for kinematics | Side-view test asset: `assets/videos/video_07_side_jump_shot.mp4` |

---

### 3. Knee, Hip & Shoulder Angles vs Accuracy (Mid-Range)

**Citation:** Effect of Knee, Hip, and Shoulder Joint angles on shooting accuracy inside the Arc in Basketball Players (2025). *Journal of Sports Biomechanics.*

**DOI:** [10.33545/26647710.2025.v7.i2f.191](https://doi.org/10.33545/26647710.2025.v7.i2f.191)

**Key findings applied in Swichy:**

| Finding | Swichy implementation |
|---------|-------------------------|
| Greater shoulder angle at release → steeper trajectory, more makes | `shoulder_alignment_lift` renamed to shoulder elevation; min 65° |
| Optimal knee + hip flexion band for mid-range | Reinforces loading-phase knee/hip rules |
| Integrated lower + upper body mechanics predict accuracy | Phase FSM sequences leg drive → ball lift → release |

---

### 4. Kinematic Sequence & Joint Work (Free Throws)

**Citation:** Relationship between kinematic sequence timing and upper extremity joint work in basketball free-throw shooting. *ISBS Proceedings* (2023).

**URL:** [NMU Commons](https://commons.nmu.edu/cgi/viewcontent.cgi?article=2804&context=isbs)

**Key findings applied in Swichy:**

| Finding | Swichy implementation |
|---------|-------------------------|
| Proficient shooters: earlier pelvis peak velocity, later elbow peak | Phase order: `loading` → `knee_flexion` → `ball_lift` → `jump` → `release` |
| Lower shoulder + elbow joint work in proficient shooters | Rules emphasize efficiency (full extension once, not multiple pushes) |
| Proximal-to-distal sequencing (knee → pelvis → elbow) | FSM dwell + hysteresis prevent premature phase jumps |

---

### 5. Standard Motion Model of the Set Shot

**Citation:** Standard motion model analysis of basketball set shot (collegiate players). *ISBS Proceedings.*

**URL:** [NMU Commons](https://commons.nmu.edu/cgi/viewcontent.cgi?article=1739&context=isbs)

**Key findings applied in Swichy:**

| Finding | Swichy implementation |
|---------|-------------------------|
| Abrupt elbow + wrist extension just before release | `release` phase: elbow > 158°, index alignment > 155° |
| Low coefficient of variation in wrist/elbow near release | Index finger landmark + `index_align` angle chain |
| Set-shot path skips jump | `ball_lift` → `release` transition for set shots |

---

### 6. Classical Reference — Shooting Mechanics

**Citation:** Knudson, D. (1993). *Biomechanics of the basketball jump shot — six key teaching points.* Journal of Physical Education, Recreation & Dance.

**Applied teaching points in Swichy coaching copy:**

1. Staggered foot base → tracked indirectly via ankle baseline / landing balance
2. Bent knees at start → `knee_flexion_loading`
3. Elbow above eyebrow, elbow in → `elbow_slot_ball_lift`
4. Extend legs then elbow → phase sequence
5. Snap wrist on release → `index_alignment_release`, `follow_through_index`
6. Hold follow-through → `follow_through_elbow`, `follow_through_index`

---

## Index Finger — Why We Track It

MediaPipe provides `left_index` / `right_index` landmarks (indices 19/20). Swichy computes an **index alignment angle** (elbow → wrist → index) and uses index vertical velocity for:

| Phase | Signal |
|-------|--------|
| `ball_lift` / `jump` | Index tracks wrist path into set point |
| `release` | Index drives upward through ball (`release_index_up_velocity`) |
| `follow_through` | Index alignment > 160° (gooseneck finish) |

This matches the ISBS set-shot finding that wrist and elbow extension are abrupt and tightly coupled at release.

---

## Rule Summary Mapped to Literature

| Swichy rule | Primary source |
|-------------|----------------|
| `knee_flexion_loading` | Jovanović 2022, 2023; mid-range accuracy study 2025 |
| `hip_hinge_loading` | ISBS kinematic sequence 2023 |
| `elbow_slot_ball_lift` | Jovanović 2022; Knudson 1993 |
| `shoulder_alignment_lift` | Mid-range accuracy study 2025 |
| `trunk_posture` | Jovanović 2022 (3PT erect torso) |
| `index_alignment_release` | ISBS set-shot model |
| `elbow_extension_release` | Jovanović 2022; MDPI Sports 2023 |
| `release_height` | Jovanović 2022 |
| `follow_through_index` | Knudson 1993; ISBS set-shot model |
| `landing_balance` | General balance / symmetry principle |

---

## Camera Setup Recommendations (from literature)

Most cited studies used **sagittal (side) view** at 60–120 fps. For best Swichy results:

- Place camera perpendicular to the shooting shoulder (side view)
- Capture full body including feet and follow-through hand
- Avoid heavy backlighting (hurts index/wrist visibility scores)
- 30 fps minimum; 60+ fps improves phase hysteresis accuracy

---

## Future Work (not yet implemented)

- **Distance-aware rule profiles** (2PT vs 3PT release angle adjustments)
- **Kinematic sequence timing** (time from knee peak to elbow peak)
- **Made/miss outcome fusion** (see `docs/PHASE_6_BALL_AND_OUTCOME.md`)
- **Joint work / efficiency metrics** from ISBS 2023

---

## Full Reference List

1. Jovanović, M., et al. (2022). Differences in Biomechanical Characteristics between Made and Missed Jump Shots. *Biomechanics*, 2(3), 428–441. https://doi.org/10.3390/biomechanics2030028
2. Jovanović, M., et al. (2023). Impact of Distance and Proficiency on Shooting Kinematics. *Sports*, 7(4), 78. https://doi.org/10.3390/sports7040078
3. (2025). Effect of Knee, Hip, and Shoulder Joint angles on shooting accuracy inside the Arc. https://doi.org/10.33545/26647710.2025.v7.i2f.191
4. ISBS (2023). Kinematic sequence and upper extremity joint work in free-throw shooting. https://commons.nmu.edu/cgi/viewcontent.cgi?article=2804&context=isbs
5. ISBS. Standard motion model of basketball set shot. https://commons.nmu.edu/cgi/viewcontent.cgi?article=1739&context=isbs
6. Knudson, D. (1993). Biomechanics of the basketball jump shot. *JOPERD*.
