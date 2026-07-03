# Phase 5 — Scoring & Coaching Feedback

Aggregates per-frame rules into one **shot score**, **coaching tips**, and a **performance plan** (drills + action items).

---

## Modules

| File | Role |
|------|------|
| `feedback/shot_tracker.py` | Shot start/end, mid-entry detection |
| `feedback/scorer.py` | Weighted 0–100 score |
| `feedback/generator.py` | Coaching tip strings |
| `feedback/performance_plan.py` | Drills, next-rep focus, capture notes |
| `feedback/console.py` | Terminal output |
| `config/scoring.yaml` | Weights and grade messages |

---

## Shot Boundaries

### Normal start
```
ready_stance → loading  OR  ready_stance → ball_lift
```

### Mid-entry start (video/live begins mid-rep)
```
First frame already in: loading | knee_flexion | ball_lift | jump | release | follow_through | landing
→ shot_in_progress = True immediately
→ started_mid_phase = True
→ entry_phase recorded
```

Also triggers if first shot skips load (e.g. `ready_stance` → `jump` on frame 2).

### Shot end
```
landing → ready_stance   (normal)
session ends mid-shot  → finalize_in_progress() with ended_early=True
```

---

## Scoring

1. **Aggregate rules** — if a rule failed on **any** frame, it fails for the shot
2. **Weight by severity** — error=3, warning=2, info=1 (`scoring.yaml`)
3. **Score** = `100 × earned_weight / total_weight`
4. **Grade** — Excellent ≥90, Good ≥75, Fair ≥60, else Needs Work

Rules for phases not captured (mid-entry) are not evaluated.

---

## `ShotSummary` Fields

```python
@dataclass
class ShotSummary:
    shot_number: int
    score: int
    passed_count, total_count: int
    passed_rules, violations: List[RuleResult]
    coaching_tips: List[str]
    phases_seen: List[str]
    started_mid_phase: bool
    ended_early: bool
    entry_phase: Optional[str]
    missing_phases: List[str]
    capture_note: str
    next_rep_focus: List[str]      # top 1-2 fixes
    practice_drills: List[str]     # mapped from rule_id
    performance_actions: List[str]
    grade: str                     # property
```

---

## Performance Plan

`performance_plan.py` maps each `rule_id` to a drill, e.g.:

| Rule | Drill |
|------|-------|
| `elbow_extension_release` | Extension pause at top of shot |
| `follow_through_index` | Gooseneck hold 2 seconds |
| `knee_flexion_loading` | Pause at bottom of load |

Also generates capture warnings for mid-entry and early-end shots.

---

## Live Feedback

### On-screen HUD
- `SHOT IN PROGRESS` + capture warning if mid-entry
- Smoothed phase, angles, violations (`hud_display.py`)
- Summary card ~3s after each shot

### Console
```
SHOT #1 — GOOD (78/100)
Capture: Recording started mid-shot at Jump...
Next rep focus:
  * Extend elbow fully at release
Drills:
  > Extension pause: at the top of each shot...
```

---

## Tuning `config/scoring.yaml`

| Setting | Effect |
|---------|--------|
| `severity_weights` | Impact of each severity on score |
| `max_coaching_tips` | Tips in generator |
| `summary_display_frames` | Overlay duration (see also `display.yaml`) |
| `score_messages` | Text per grade band |

---

## Pipeline Position

```
BiomechanicsEngine (per frame)
    ↓
ShotTracker (boundaries + mid-entry)
    ↓
score_shot() → generate_coaching_tips() → build_shot_performance_plan()
    ↓
HUD + Console + SessionRecorder → PDF
```

---

## Related

- [REPORTING.md](REPORTING.md)
- [PHASES_OVERVIEW.md](PHASES_OVERVIEW.md)
- [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md)
