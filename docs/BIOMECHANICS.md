# Biomechanical Rule Engine

## What Changed

| File | Change |
|------|--------|
| [`analysis/engine.py`](../analysis/engine.py) | **New** — `BiomechanicsEngine` |
| [`analysis/models.py`](../analysis/models.py) | **New** — `RuleResult`, `AnalysisResult` |
| [`config/biomechanics.yaml`](../config/biomechanics.yaml) | **New** — 10 configurable rules |
| [`pipeline.py`](../pipeline.py) | Runs rule engine after phase detection |
| [`visualization/renderer.py`](../visualization/renderer.py) | Shows rule pass/fail on screen |

---

## Why It Changed

Phase 3 answers **when**. Phase 4 answers **whether form is acceptable** at that moment.

We use **ranges** instead of exact targets because:
- Body proportions vary between players
- Camera distance affects world-coordinate scale slightly
- Good form is a **zone**, not a single angle (e.g. knee 70–130°, not "exactly 90°")

---

## How Evaluation Works

```
For each frame:
  1. Get current phase from Phase 3
  2. Load rules from biomechanics.yaml
  3. Skip rules whose `phases` list does not include current phase
  4. Measure metric (angle, velocity, or height)
  5. Compare to min/max range
  6. Return RuleResult (passed, severity, message)
```

### Example rule

```yaml
elbow_extension_release:
  name: "Elbow Extension (Release)"
  metric: elbow
  phases: [release]
  min: 155
  max: 180
  severity: warning
  message_pass: "Full extension at release"
  message_fail: "Extend elbow fully at release for arc and power"
```

At `release` phase, if `right_elbow` angle is 148° → **failed**, message shown in orange.

---

## Available Metrics

| Metric | Source | Used for |
|--------|--------|----------|
| `knee` | 3D knee angle | Knee flexion |
| `hip` | 3D hip angle | Hip hinge |
| `elbow` | 3D elbow angle | Elbow slot, extension, follow-through |
| `shoulder` | 3D shoulder angle | Shoulder alignment |
| `trunk` | Trunk vs vertical | Posture |
| `head_velocity` | Nose Y velocity | Head stability |
| `release_height` | Wrist Y − nose Y | Release point height |
| `ankle_rise` | Ankle Y − baseline | Landing balance |

---

## RuleResult Structure

```python
@dataclass
class RuleResult:
    rule_id: str
    name: str
    passed: bool
    severity: str       # info | warning | error
    message: str
    phase: str
    measured_value: float
    min_value: float
    max_value: float
```

---

## Severity Levels

| Level | Color on screen | Meaning |
|-------|-----------------|---------|
| `info` | Yellow | Suggestion, minor adjustment |
| `warning` | Orange | Important form issue |
| `error` | Red | Major mechanical problem (reserved for future strict rules) |

---

## Why Ranges, Not Exact Values

Basketball coaching literature describes **zones**:
- "Elbow at or above shoulder at set point" — not 90.0°
- "Full extension at release" — 155–180°, not exactly 180°

Exact values fail on valid variations (tall vs short players, different shooting styles like one-motion vs two-motion).

---

## Adding a New Rule

1. Edit [`config/biomechanics.yaml`](../config/biomechanics.yaml):

```yaml
my_new_rule:
  name: "My Rule"
  metric: elbow
  phases: [ball_lift]
  min: 80
  max: 110
  severity: warning
  message_pass: "Looks good"
  message_fail: "Adjust your form"
```

2. If you need a new metric, add it to `_measure()` in [`analysis/engine.py`](../analysis/engine.py).

No code change needed for rules using existing metrics.

---

## AI Concepts to Study

### Concept: Rule-Based Expert System

**What it is:** A system that applies IF-THEN rules from domain knowledge instead of learning from data.

**Why we use it:** Basketball biomechanics has known coaching principles. Rules are interpretable — you can explain *why* feedback was given.

**Alternatives:** ML classifiers, learned thresholds, LLM coaching.

**Advantages:** No training data, explainable, tunable per skill level.

**Disadvantages:** Manual tuning, may not capture individual style.

**Difficulty:** Beginner

**Topics:** Expert systems, knowledge representation, YAML configuration

---

### Concept: Phase-Conditioned Evaluation

**What it is:** Rules only apply in specific contexts (phases).

**Why we use it:** The same angle can be good in one phase and bad in another.

**Difficulty:** Beginner (once phases exist)

---

## Pipeline Position

```
Phase Detection
    ↓
Biomechanical Rules  ← YOU ARE HERE
    ↓
Visualization (pass/fail overlay)
    ↓
Phase 5: Shot scoring (planned)
```

See also: [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md), [PHASE_DETECTION.md](PHASE_DETECTION.md)
