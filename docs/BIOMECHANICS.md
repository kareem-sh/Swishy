# Biomechanical Rule Engine

Phase-aware rule evaluation: **when** (from FSM) + **whether** form is in range (from YAML).

---

## Modules

| File | Role |
|------|------|
| `analysis/engine.py` | `BiomechanicsEngine` |
| `analysis/models.py` | `RuleResult`, `AnalysisResult` |
| `config/biomechanics.yaml` | 12 rules |
| `docs/BIOMECHANICS_RESEARCH.md` | Papers → rule rationale |

---

## How Evaluation Works

```
For each frame:
  1. Get current phase
  2. Load rules from biomechanics.yaml
  3. Skip rules whose phases[] does not include current phase
  4. Measure metric (angle, velocity, height)
  5. Compare to min/max range
  6. Return RuleResult (passed, severity, message)
```

If metric is unavailable (occluded landmark) → rule skipped for that frame.

---

## The 12 Rules

| Rule ID | Phase(s) | Checks |
|---------|----------|--------|
| `knee_flexion_loading` | loading, knee_flexion | Knee 70–125° |
| `hip_hinge_loading` | loading, knee_flexion | Hip 145–175° |
| `elbow_slot_ball_lift` | ball_lift | Elbow 75–115° |
| `shoulder_alignment_lift` | ball_lift, jump | Shoulder 65–125° |
| `trunk_posture` | stance → release | Trunk lean 5–22° |
| `head_stability` | loading → release | Nose velocity ≤ 0.08 m/s |
| `index_alignment_release` | release | Index align 155–180° |
| `elbow_extension_release` | release | Elbow 158–180° |
| `release_height` | release | Wrist above nose 0.18–0.55 m |
| `follow_through_elbow` | follow_through | Elbow 155–180° |
| `follow_through_index` | follow_through | Index align 160–180° |
| `landing_balance` | landing | Ankle rise ≤ 0.06 m |

Ranges are **zones**, not exact targets — see [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md) for why.

---

## Available Metrics

| Metric | Source |
|--------|--------|
| `knee`, `hip`, `elbow`, `shoulder` | 3D joint angles (shooting side) |
| `index_align` | elbow → wrist → index angle |
| `trunk` | Hip-mid to shoulder-mid vs vertical |
| `head_velocity` | abs(nose vertical velocity) |
| `release_height` | wrist_y − nose_y |
| `ankle_rise` | ankle_y − standing baseline |

---

## Adding a Rule

1. Add entry to `config/biomechanics.yaml`
2. Add drill mapping in `feedback/performance_plan.py` → `RULE_DRILLS`
3. If new metric needed, extend `_measure()` in `analysis/engine.py`

---

## Severity

| Level | Meaning |
|-------|---------|
| `info` | Minor suggestion |
| `warning` | Important form issue |
| `error` | Major issue (reserved for strict rules) |

Weights in `scoring.yaml` affect shot score.

---

## Related

- [PHASE_DETECTION.md](PHASE_DETECTION.md)
- [FEEDBACK_SCORING.md](FEEDBACK_SCORING.md)
- [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md)
