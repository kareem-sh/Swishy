# Swichy Product Roadmap

**Goal:** Mobile app where a player films or uploads shots → gets form feedback, make/miss, stats, and coaching for **jump shot, layup, and fadeaway**.

**Today:** Python prototype works (pose, phases, rules, PDF). **Next:** ball tracking, player stats, mobile app.

**Also read:** [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md) · [PHASE_6_BALL_AND_OUTCOME.md](PHASE_6_BALL_AND_OUTCOME.md)

---

## At a glance

| | |
|---|---|
| **Final app** | Live camera + upload video · IN/OUT · 3 shot types · stats · best shots · phase TTS |
| **AI approach** | Rules (explainable) + small ML models (adapt to player) — not rules-only forever |
| **Time to mobile MVP** | ~3–4 months full-time · ~5–6 months part-time (4–6 people) |
| **Best dataset** | **Your own phone videos** — public data only for bootstrapping |

---

## 1. What you are building

```
Phone camera / uploaded video
        ↓
   Pose (MediaPipe)  +  Ball/Rim (YOLO)
        ↓
   Shot type: jump | layup | fadeaway
        ↓
   Phases + Form score  +  Make/Miss
        ↓
   Coaching (form ≠ outcome)  +  Stats  +  TTS
```

**Four things to measure (keep separate):**

| | Question |
|---|----------|
| **Form** | Was technique good for this shot type? |
| **Outcome** | Did it go in? |
| **vs personal best** | Better or worse than their usual? |
| **Coach message** | Depends on form + outcome together |

Good form can miss. Bad form can still go in. The app must say both honestly.

---

## 2. Datasets — which to use (ranked)

> **No public dataset has everything** (phone camera + layup + fadeaway + MediaPipe + make/miss).  
> Use public data to **start**, then **record your own gym clips** for real quality.

### Tier 1 — Use these first ✅

| Priority | Dataset | Link | Use for | Size |
|:--:|---------|------|---------|------|
| **#1** | **Your own videos** | Record on target phones | Everything that matters | Start with 50 clips, grow to 500+ |
| **#2** | **Roboflow: Hoop + Ball + Player** | [Download](https://universe.roboflow.com/basketball-stat-tracker/basketball-hoop-ball-and-player) | Train YOLO ball/rim detector | ~199 images |
| **#3** | **Roboflow: basketball-yolo-dataset** | [Download](https://universe.roboflow.com/yolo-train-rqswv/basketball-yolo-dataset-hpwha) | Extra ball/hoop/player labels | ~321 images |
| **#4** | **MediaPipe Pose** | [Docs](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) | Pose on mobile (already in Swichy) | Built-in model |

**Action:** Download Roboflow #2 + #3 → fine-tune YOLOv8n → then re-train on 200+ frames from **your phones**.

---

### Tier 2 — Helpful, not enough alone ⚠️

| Dataset | Link | Good for | Why not enough |
|---------|------|----------|----------------|
| **Roboflow rim set** | [449 images](https://universe.roboflow.com/cv-research-a09ll/basketball-detection-sr1wz) | Rim detection | Fixed angles, not phone POV |
| **Basketball-51** | [Paper](https://www.researchgate.net/publication/352155386) | Make/miss labels (10k clips) | TV broadcast, no layup/fadeaway, no pose |
| **HoopLab** | [Paper](https://doi.org/10.5121/csit.2025.152402) | Mobile Flutter + YOLO reference | Study their approach, not their data |
| **Ball trajectory code** | [GitHub](https://github.com/AlvinYapAbidin/BallTrajectoryPrediction) | YOLO + Kalman example | Code reference only |

---

### Tier 3 — Research only (skip for MVP) 📚

| Dataset | Link | Notes |
|---------|------|-------|
| DeepSport Instants | [Kaggle](https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset) | Pro arena, fixed cameras — not phone |
| PoseShot | [Paper](https://doi.org/10.1038/s41598-026-41025-0) | 75 free throws only |
| SpaceJam joints | [Paper](https://doi.org/10.21203/rs.3.rs-2947413/v1) | Old 2D joints, not MediaPipe |

---

### What you must record yourself

Label every clip with:

| Field | Values |
|-------|--------|
| `shot_type` | jump_shot · layup · fadeaway |
| `outcome` | make · miss |
| `camera_view` | side · front · behind_hoop |
| `player_id` | for stats |

**Minimum targets (fast team):**

| Need | Amount |
|------|--------|
| Ball/rim boxes (fine-tune YOLO) | 200–500 frames |
| Shot type labels | 150+ clips **per type** |
| Make/miss test set | 300 makes + 300 misses |
| Full reps with phases (optional ML) | 100+ clips |

**Shot type definitions:** [PMC shot types](https://pmc.ncbi.nlm.nih.gov/articles/PMC4454648/) · Swichy rules: [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md)

---

## 3. Step-by-step plan (do in this order)

### Step 0 — Setup (Week 1–2)

- [ ] Everyone reads [MANUAL_COMPLETION_GUIDE.md](MANUAL_COMPLETION_GUIDE.md)
- [ ] Create labeling spreadsheet (columns above)
- [ ] Record **50 clips** on phones (mix of jump / layup / make / miss)
- [ ] Agree shared data types in `ball/models.py`, `player/models.py` (see §6)

---

### Step 1 — Ball + make/miss (Week 3–6) 🎯 Priority

**Owner:** Person A · **Files:** `ball/*`

1. Download Roboflow datasets (#2, #3)
2. Train YOLOv8n (`pip install ultralytics`)
3. Track ball with Kalman filter
4. Set hoop ROI (manual `hoop_roi.yaml` or auto from rim box)
5. Decide IN/OUT from ball path vs hoop
6. Show **IN / OUT** on video overlay

**Done when:** 80%+ correct on 50 labeled team clips.

**PyTorch?** Only via Ultralytics for training. Runtime = ONNX on mobile later.

---

### Step 2 — Save player stats (Week 4–7)

**Owner:** Person D · **Files:** `player/*`

1. SQLite schema (players, sessions, shots)
2. Hook `SessionRecorder` → save every shot after video/live
3. Show: make %, reps per session, form score average
4. Flag **best shots** (make + high form score)

**Done when:** After running `main.py`, stats appear in DB.

---

### Step 3 — Phase TTS (Week 5–6)

**Owner:** Person E · **Files:** `coaching/tts.py`

1. Map phase → short phrase (“Loading”, “Release”, “Follow through”)
2. Speak on phase change (debounce like HUD hold)
3. Use `edge-tts` (Python) or native TTS on mobile later

**Done when:** Live session speaks phase names without spamming.

---

### Step 4 — Three shot types (Week 7–10)

**Owner:** Person B · **Files:** `phase_detection/layup.py`, `fadeaway.py`

1. Keep current FSM for **jump shot** (tune on `video_07_side_jump_shot.mp4`)
2. New phase logic for **layup** (test on `video_02_one_on_one.mp4`)
3. New phase logic for **fadeaway** (custom clips)
4. Router: detect type → pick correct phases + rules

**Done when:** App picks correct type on 70%+ of 30 labeled clips.

---

### Step 5 — Smart coaching (Week 11–14)

**Owner:** Person E · **Files:** `coaching/quadrant.py`

| Form | Result | Say |
|------|--------|-----|
| Good | Make | “Great rep — saved as reference.” |
| Good | Miss | “Form was solid — adjust aim/arc.” |
| Bad | Make | “Went in, but fix [X] before game speed.” |
| Bad | Miss | “Work on [top 2 rule fixes].” |

Extend PDF: show IN/OUT + shot type + this message.

---

### Step 6 — ML shot type (Week 10–13, optional but better)

**Owner:** Person C · **Files:** `models/shot_type/`

1. Export angle sequences from pipeline to CSV
2. Train small **1D-CNN or TCN** in PyTorch (150+ clips per type)
3. Replace heuristic router when accuracy > 75%

**PyTorch?** Yes — this is the first real training task.

---

### Step 7 — Hybrid form score (Week 14–18)

**Owner:** Person C + You

```
form = 40% rules + 40% ML model + 20% vs personal baseline
```

Start 100% rules (today). Add ML after 500+ labeled shots.

---

### Step 8 — Mobile app (Week 14–20)

**Owner:** Person F · **Stack:** Flutter + MediaPipe Tasks + ONNX

1. Camera live + video upload screens
2. MediaPipe pose on device ([Android](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/android) · [iOS](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/ios))
3. ONNX ball model overlay
4. Sync stats from Person D’s SQLite schema
5. TTS + coaching strings from Person E

**Reference app:** [HoopLab paper](https://doi.org/10.5121/csit.2025.152402) (Flutter + YOLO)

**Done when:** Beta APK with live pose + upload + make/miss + basic stats.

---

## 4. Timeline (fast team, ~15–20 h/week)

```
Week  1-2   Setup + 50 clips
Week  3-6   Ball + IN/OUT          ← start here
Week  4-7   Player stats DB
Week  5-6   Phase TTS
Week  7-10  Jump / layup / fadeaway
Week 10-13  ML shot type (PyTorch)
Week 11-14  Smart coaching + best shots
Week 14-20  Mobile MVP
Week 18-22  Shot chart + polish
```

| Milestone | Week | Check |
|-----------|------|-------|
| M1 | 6 | IN/OUT works on desktop video |
| M2 | 7 | TTS speaks phases |
| M3 | 9 | Stats saved per player |
| M4 | 10 | 3 shot types routed |
| M5 | 14 | Coaching uses form + outcome |
| M6 | 20 | Mobile beta |

---

## 5. Team — who does what (no overlap)

| Person | Owns | Weeks | Never touches |
|--------|------|-------|---------------|
| **A — Ball** | `ball/*` | 1–16 | pose, DB, mobile |
| **B — Phases** | `phase_detection/*` | 1–14 | ball, mobile |
| **C — ML/Data** | `models/`, labeling | 3–16 | pipeline merge, mobile |
| **D — Stats** | `player/*`, SQLite | 2–18 | CV, TTS |
| **E — Coaching** | `coaching/*`, reports | 4–16 | ball, mobile |
| **F — Mobile** | `mobile/` Flutter | 10–22 | training, rules YAML |
| **You** | `pipeline.py`, integration | all | — |

**Rule:** Modules talk only through shared types (§6). No one edits another person’s folder.

---

## 6. Shared data (integration contract)

```python
# ball/models.py
BallDetection(frame, x, y, w, h, confidence)
OutcomeResult(made: bool | None, confidence, reason)

# player/models.py
ShotRecord(shot_type, form_score, outcome, court_x, court_y, is_best_shot)

# coaching/models.py
CoachingMessage(text, tts_text)
```

You merge these in `pipeline.py` and `ball/fusion.py`.

---

## 7. Tech stack (simple)

| Job | Tool | Need PyTorch? |
|-----|------|---------------|
| Pose | MediaPipe (lite on phone) | No |
| Ball/rim | YOLOv8n → ONNX | Yes, for training only |
| Shot type | Small TCN on angles | Yes |
| Make/miss v1 | Ball path + hoop geometry | No |
| Stats | SQLite | No |
| TTS | edge-tts / flutter_tts | No |
| Mobile | Flutter | No |
| Coaching text | Templates (+ LLM later) | No |

**Python repo** = train + test algorithms. **Mobile** = run exported models.

---

## 8. Shot types (quick reference)

| Type | Camera | Phases differ? | Ball needed? |
|------|--------|----------------|--------------|
| Jump shot | Side view best | Current 8-phase FSM | Nice to have |
| Layup | Side / behind hoop | Shorter: approach → release → land | **Yes** (near rim) |
| Fadeaway | Side view | Jump shot + trunk lean back | Nice to have |

Do **not** force layups into the jump-shot FSM. Separate files per type.

---

## 9. Links (study when needed)

**Datasets**
- Roboflow hoop/ball: https://universe.roboflow.com/basketball-stat-tracker/basketball-hoop-ball-and-player
- Roboflow alt: https://universe.roboflow.com/yolo-train-rqswv/basketball-yolo-dataset-hpwha
- Basketball-51: https://www.researchgate.net/publication/352155386

**Tools**
- YOLOv8: https://docs.ultralytics.com
- MediaPipe mobile: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- PyTorch: https://pytorch.org/tutorials/
- edge-tts: https://github.com/rany2/edge-tts

**References**
- HoopLab mobile app: https://doi.org/10.5121/csit.2025.152402
- Ball tracking code: https://github.com/AlvinYapAbidin/BallTrajectoryPrediction
- Shot type definitions: https://pmc.ncbi.nlm.nih.gov/articles/PMC4454648/
- Swichy biomechanics: [BIOMECHANICS_RESEARCH.md](BIOMECHANICS_RESEARCH.md)

---

## 10. This week — start here

| Who | Task |
|-----|------|
| **Everyone** | Read MANUAL_COMPLETION_GUIDE + record 10 phone clips |
| **A** | Download Roboflow #2, train first YOLO |
| **B** | Tune jump-shot phases on `video_07_side_jump_shot.mp4` |
| **C** | Create labeling spreadsheet |
| **D** | Draft SQLite schema in `player/` |
| **E** | Write form×outcome message table |
| **You** | Add `outcome` + `shot_type` fields to `ShotSummary` plan |

---

## 11. Common mistakes to avoid

| Mistake | Fix |
|---------|-----|
| Expect one dataset to do everything | Tier 1 public + your own clips |
| Only side camera for make/miss | Use behind-hoop view for IN/OUT; side for form |
| One score for form and result | Keep four scores separate (§1) |
| One FSM for all shot types | Separate layup / fadeaway logic |
| Train ML before labeling data | Steps 1–5 first; ML at step 6 |
| Everyone edits `pipeline.py` | One integration owner (you) |

---

*Swichy Phases 1–5b complete · Phase 6 stubs in `ball/` · Updated 2026*
