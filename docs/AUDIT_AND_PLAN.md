# Swichy — Pre-Implementation Audit & Plan

**Branch:** `pose-edits` · **Date:** 9 August 2026 · **Status:** PLAN ONLY — no code modified
**Companion documents:** [`resources.md`](../resources.md) (evidence map) · [`research.md`](../research.md) (literature review, v1 — requires the corrections in §1)

> **Scope reality check.** Parts 29, 41–43 of the brief (retune against real shots with labelled ground truth; multi-player height-fairness regression) are **BLOCKED**. The repo has 7 test videos and 5 images, but **no recorded player heights, no labelled shot dataset, and no coach-labelled ground truth**. Everything else in the brief is addressed below.

---

## 1. RESEARCH AUDIT

### 1.1 Reframing (Part 1)

The research question is rebuilt around **teaching shooting technique in practice**. Removed from scope: defenders, game context, tactical situations, game-performance prediction, make/miss prediction.

Two papers previously in scope are now **out of primary scope** and retained only as background: *video_02_one_on_one*-style contested-shot literature, and "Biomechanical Adjustments of the Jump Shot Over Differently High Opponents".

### 1.2 🚨 Two corrections to `research.md` v1

The council rejected two of my earlier conclusions. Both rejections are correct.

**CORRECTION 1 — "Elbow set point is invariant" was overstated.**
v1 presented six comparisons as convergent evidence. Five of those are **non-significant p-values** (0.471, 0.706, 0.228) from n=10–34 samples, **all from one research group**. Failing to reject H₀ at that sample size is a **null-hypothesis fallacy**, not evidence of invariance. No power analysis exists in any of the papers. Motor-learning theory (Bernstein's degrees-of-freedom problem) actively predicts functional variability instead.
→ Reclassified `DIRECTLY SUPPORTED` → **`ENGINEERING DECISION`** (a tunable prior).

**CORRECTION 2 — "Deeper knee load discriminates proficiency" is not defensible.**
Four findings, three directions (see [`resources.md`](../resources.md) §D3). One says deeper is better (p<0.001, 3-point only), two say no difference, one says the **opposite**.
→ Reclassified **`NOT SUPPORTED`**. `knee_flexion_loading` must be demoted from scored rule to displayed metric.

### 1.3 Claim classification (Part 12)

| # | Claim | Class | Action |
|---|---|---|---|
| 1 | Elbow set point stable across distance | `ENGINEERING DECISION` | Keep as prior, relabel rationale |
| 2 | Deeper knee load = proficiency | `NOT SUPPORTED` | **Demote to metric** |
| 3 | Release height discriminates; release angle doesn't | `SUPPORTED WITH LIMITATIONS` | Keep; **must state R²=0.005–0.012 out loud** |
| 4 | Ball launch angle 50–55° | `SUPPORTED WITH LIMITATIONS` | **Currently unmeasurable — see §10** |
| 5 | Height-normalise release height | `SUPPORTED WITH LIMITATIONS` | Implement; single-group precedent |
| 6 | Vision height unreliable for scoring | `DIRECTLY SUPPORTED` (physics) | Confirm; argue from geometry not journals |
| 7 | Vertical displacement separates set/jump | `INFERRED` | Implement **with confidence band, not binary** |
| 8 | Evidence base adequate | `SUPPORTED WITH LIMITATIONS` | Disclose single-lab dependency |

### 1.4 Findings that survive scrutiny

1. **Ball launch angle 50–55°** — three independent methodologies (empirical 3D cine, physics simulation, high-speed ball tracking), 29 years apart, Tier-1 journals. Strongest number in the corpus.
2. **Release height (÷ body height) > release angle** as a proficiency correlate — two independent groups.
3. **Set vs jump shot are mechanically distinct** — Okubo & Hubbard's shoulder-vertical-velocity definition, plus a ~2× vertical-displacement separation.
4. **External focus of attention beats internal focus** for motor learning — the most replicated finding, and it **contradicts Swichy's current feedback wording** (§1.5).
5. **Monocular scale ambiguity is fundamental** — argued from geometry, not citation count.

### 1.5 🚨 New finding: our feedback wording is the less-effective kind

Council peer review flagged motor-learning pedagogy as a universal blind spot. Research confirms: **external focus** (on the movement's effect) outperforms **internal focus** (on body parts) in retention and transfer, including basketball free throws. Mechanism: the constrained-action hypothesis.

`config/biomechanics.yaml` messages are almost entirely **internal focus** — "Bend knees deeper", "Tuck elbow under the ball", "Stay taller through the torso". These are the form the literature says is worse. Rewrite required (§13, Phase 12).

---

## 2. SOURCE QUALITY AUDIT

| Tier | Journals in corpus | Status |
|---|---|---|
| **Tier 1** (Scopus + WoS SCIE + MEDLINE) | *J. Sports Sciences*, *J. Biomechanics*, *Sports Biomechanics* (ISBS), *PLOS Comput Biol*, *PLoS ONE*, *J. Human Kinetics* | ✅ Strongest citations |
| **Tier 2** (Scopus + WoS ESCI, PubMed/PMC) | *Frontiers Sports Act Living*, *JFMK*, *Sensors*, *PeerJ* | ⚠️ Disclose ESCI status |
| **Tier 3** | *Biomechanics* (MDPI), *Proceedings* (ISEA conf.) | ⚠️ Cite for data/model only, disclose venue |
| **REJECTED** | IJERPH (WoS-delisted Feb 2023), physicaleducationjournal.net, iosrjournals.org | ❌ See `resources.md` §D5 |

### The single biggest credibility risk
**Single-lab dependency.** Most numeric shooting-kinematics data comes from the Cabarkapa/Fry group (University of Kansas), in ESCI-only journals, n=10–34, often *recreational* participants. There is **no independent replication**. A jury that notices this before we disclose it is the worst outcome; disclosing it first converts a weakness into evidence of rigour.

### Verification status
7 sources ✅ full-text verified · 8 ⚠️ abstract-only · 2 ❌ unverified (see `resources.md` Part F). **Abstract-only numbers are never presented as full-text verified.**

---

## 3. `resources.md` PLAN

**Created.** [`resources.md`](../resources.md) is live with: Part A core sources (9), Part B height/anthropometry (3), Part C motor learning, Part D contested/corrected/rejected, Part E source→rule→code map, Part F 9 open verification items, Part G 5 self-named research gaps.

**Maintenance rule:** every claim in `research.md` or this document must resolve to an entry in `resources.md`. No claim ships without a class and a verification marker.

---

## 4. PLAYER HEIGHT RESEARCH AUDIT

### 4.1 What the research says

| Question | Answer | Class |
|---|---|---|
| Does height affect release height? | Yes — mechanically necessary and universally reported | `DIRECTLY SUPPORTED` |
| Does height affect optimal *launch angle*? | Yes, but **geometrically, via release height** — not as an independent biological factor. Taller ⇒ higher release ⇒ lower optimal angle (~52.2° at 5'4" → ~48.7° at 7'0") | `INFERRED` (primary source unverified — `resources.md` F2) |
| Does height affect joint angles? | **No evidence found.** | `NOT SUPPORTED` |
| Should release height be normalised by body height? | Yes — consistent precedent across 4 Cabarkapa papers, which report it *only* as a ratio | `SUPPORTED WITH LIMITATIONS` |
| Should height influence *scoring*? | Only via normalisation of height-dependent metrics — never via height-conditioned angle targets | `ENGINEERING DECISION` |

### 4.2 Height-dependent vs height-independent metrics

| Metric | Height-dependent? | Representation | Why |
|---|---|---|---|
| Release height | **Yes** | `release_height / player_height` | Literature precedent (A2); raw metres are incomparable across players |
| Jump height / vertical displacement | **Partly** | Report **absolute** primarily | Jump height is an athletic capacity, not a body-scale artefact. Normalising it would mask real ability differences |
| Set/jump classification threshold | **Unknown** | Start **absolute**, log both | 15.3 vs 26.9–31.2 cm gap is wide relative to plausible height scaling. Revisit with data |
| Knee / hip / elbow / shoulder angles | **No** | Absolute degrees | Angles are dimensionless ratios of segment orientation; no evidence of height dependence |
| Trunk lean | **No** | Absolute degrees | Same reasoning |
| Wrist-relative-to-nose (current `release_height` metric) | **No — already normalised** | Keep as-is | This is a *body-relative* measure. It is implicitly height-invariant, which is a hidden strength of the current design |
| Ball launch angle | **No** (but its *optimum* is) | Absolute degrees vs computed target | See §4.3 |

### 4.3 The important structural insight

Do **not** build a height→target lookup table. The height effect on optimal launch angle is **geometry**, computable per shot:

```
optimal_launch_angle ≈ 45° + 0.5 × arctan( (3.05 − release_height_m) / horizontal_distance_m )
```

This personalises without inventing biology, and degrades gracefully — with no height, no ball tracking, and no hoop distance, it simply isn't offered.
⚠️ **Blocked:** Swichy measures neither ball launch angle nor horizontal hoop distance today.

### 4.4 Vision-estimated height — recommendation

**Do not use vision height for scoring.** Reasoning is geometric, not bibliographic:
- Monocular projection creates fundamental size/distance ambiguity.
- MediaPipe world landmarks come from fitting the **GHUM statistical body model** — they carry anatomical priors, so estimates regress toward population means.
- The best MediaPipe height result found (166 subjects, **1.54 ± 0.64%** error) **required a physical reference object** in frame. Swichy has none.

**Recommendation:** user-typed height is authoritative. Vision height may later serve as a **sanity check** (flag a >15% discrepancy for re-entry), never as a silent override, and never as a scoring input until validated against known-height players.

### 4.5 UX risk the council caught
"What happens when the kid lies, is 11 and doesn't know, or types cm when the app expects inches?" — Mitigations: explicit unit selector, plausibility range (120–230 cm) with a soft warning outside it, `height_source` recorded, and **height always optional**.

---

## 5. COORDINATE AUDIT

### 5.1 Ground truth established (Part 23)

Real MediaPipe run on `assets/test.jpg` and 3 videos (1,821 frames):

```
nose.y         = -0.453      ankle_mid.y = +0.466
shoulder_mid.y = -0.441      hip_mid.y   = -0.001
```

**`pose_world_landmarks` are Y-DOWN.** Confirmed on 1821/1821 frames. The docs and code assume Y-UP in three places: [`geometry/vectors.py:60`](../geometry/vectors.py#L60), [`pose/landmarks.py:100`](../pose/landmarks.py#L100), [`docs/ANGLES_3D.md:33`](ANGLES_3D.md#L33).

> Note: Google's official Pose Landmarker page does **not** document axis directions, origin, or units for world landmarks. Our empirical measurement is therefore the authoritative reference — and that fact is worth stating in the thesis.

### 5.2 Measured runtime impact

| Metric | Measured | Rule window | Result |
|---|---|---|---|
| `trunk` | 165.4–179.8° (mean 170–174°) | `[5, 22]` | **0 / 1359 frames pass** |
| `release_height` (`wrist_y − nose_y`) | negative when wrist above nose | `[0.18, 0.55]` | **0% pass at release** (videos 1 & 7) |
| `ankle_rise` (jump signal) | real jump = **−0.118** | fires if `> +0.03` | **Never fires correctly** |
| `knee_flexion_loading` | — | `[70, 125]` | **0–9% pass** |

True trunk lean is `180 − measured` ≈ **5.6–14°**, which *would* pass. Every well-executed shot is being penalised.

### 5.3 Blast radius — precisely scoped

**✅ UNAFFECTED (do not touch):** all six three-landmark angles — elbow, knee, hip, shoulder, ankle_flexion, index_align. `angle_between_vectors` measures the angle between two segment vectors; a whole-frame reflection is orthogonal and preserves it.

**❌ AFFECTED:**

| Site | Defect |
|---|---|
| `geometry/vectors.py::angle_from_vertical` | `up = [0,1,0]` points at the floor → returns `180 − true` |
| `angles/calculator.py::_compute_trunk_angle` | Only consumer of the above |
| `analysis/engine.py::_measure` | `release_height` sign inverted; `ankle_rise` sign inverted |
| `phase_detection/features.py` | All `_lm_y` semantics inverted |
| `phase_detection/detector.py` | Every vertical comparison inverted (table below) |
| `pipeline.py::_resolve_shooting_side` | Picks the **lower** wrist; default is `"auto"`, so live |
| `pose/landmarks.py`, `geometry/vectors.py`, `docs/ANGLES_3D.md` | Docstrings state the wrong convention |
| `ml/datasets/*.npz` | Prebuilt features encode the broken convention |

| Detector expression | Intent | Actual under Y-DOWN |
|---|---|---|
| `wrist_y < shoulder_y + 0.05` | wrist below shoulder | wrist **above** |
| `wrist_velocity_y > 0.03` | wrist rising | wrist **descending** |
| `wrist_y > hip_y_avg` | wrist above hip | wrist **below** |
| `ankle_rise > 0.03` | feet leaving floor | feet leaving floor is **negative** |
| `hip_velocity_y < −0.02` | hips dropping | hips **rising** |
| `wrist_y > _wrist_peak_y` | track highest wrist | tracks **lowest** |

### 5.4 Decision — canonical Y-UP boundary (Parts 23–25)

**Adopt option (a): negate Y once in `pose/landmarks.py::extract_world_landmarks`.**

Council rationale (First Principles, endorsed by Executor and Contrarian): *"One coordinate system, established at ingestion, is the actual fix; the rest is patching symptoms."* Options (b) and (c) leave Y-down arrays circulating so the next contributor reintroduces the bug — exactly what happened here.

After the negation, every existing Y-UP assumption in the codebase becomes **true**, and all three docstrings become correct rather than needing rewriting.

---

## 6. PHASE AUDIT

### 6.1 Measured FSM pathology

| Video | Frames | Dominant phase | Verdict |
|---|---|---|---|
| `video_01_free_throw` | 324 | `knee_flexion` **213 (66%)** | **Jammed** |
| `video_04_shooting_alone` | 146 | `knee_flexion` **83 (57%)** | **Jammed** |
| `video_07_side_jump_shot` | 1351 | `ready_stance` 492, `ball_lift` 365 | Plausible but noisy |

**Root cause of the jam:** `knee_flexion`'s only exit requires `wrist_velocity_y > 0.03 AND wrist_y > hip_y_avg`. Under Y-DOWN, `wrist_y > hip_y_avg` means wrist **below** hip, so during the actual ball lift the condition is false and the FSM cannot advance.

### 6.2 Per-phase Y-semantics table

| Phase | Entry condition | Y-dependent? | Post-fix action |
|---|---|---|---|
| `ready_stance` | `total_velocity < 0.2` | No (magnitude) | Verify only |
| `loading` | hip drop, knee delta, wrist below shoulder | **Yes ×3** | Re-verify + retune |
| `knee_flexion` | `knee_angle_delta > 0.5` | No (angle) | Verify |
| `ball_lift` | `wrist_velocity_y > 0.03`, `wrist_y > hip_y` | **Yes ×2** | **Fix the jam**, retune |
| `jump` | `ankle_rise > 0.03` or `ankle_velocity_y > 0.02` | **Yes ×2** | Sign fix is critical |
| `release` | wrist peak, elbow ext, wrist velocity, index velocity | **Yes ×3** | Retune |
| `follow_through` | `wrist_velocity_y < −0.02`, elbow, index | **Yes ×2** | Retune |
| `landing` | ankle near baseline | **Yes** | Retune |

### 6.3 Retuning is mandatory, not optional
Council (First Principles): thresholds hand-tuned against inverted semantics *"were never tuned against reality — they're fit to noise."* The sign fix alone will not produce a correct detector.

---

## 7. TIMING AUDIT

**Good news:** velocity already uses `Δposition / Δtime` from real timestamps ([`pipeline.py::_compute_dt`](../pipeline.py#L510)). Part 30's main concern is already satisfied.

**Real problem — frame-count thresholds under variable FPS.** Measured asset FPS: **24.00, 29.99, 23.98**. `DEFAULT_FPS = 30.0` matches none of them.

| Setting | Value | At 30 fps | At 24 fps | Drift |
|---|---|---|---|---|
| `hysteresis_frames` | 2 | 67 ms | 83 ms | +25% |
| `min_dwell_frames` | 2 | 67 ms | 83 ms | +25% |
| `VISIBILITY_HOLD_FRAMES` | 5 | 167 ms | 208 ms | +25% |
| `summary_display_frames` | 60–90 | 2–3 s | 2.5–3.75 s | +25% |

At the ~15 fps the brief mentions, drift is **+100%**. **Action:** convert to milliseconds, derive frame counts from measured FPS at runtime. **Ship as a separate change** (§13).

---

## 8. CONFIG AUDIT

| File | Finding |
|---|---|
| `settings.py` | `MODEL_PATH` = **lite**, README claims **full** (§10). `DEFAULT_FPS = 30.0` inaccurate. `SHOOTING_HAND = "auto"` → the inverted detector is live |
| `biomechanics.yaml` | `trunk_posture` + `release_height` unreachable (§5.2). `knee_flexion_loading` unsupported (§1.2). `shoulder_alignment_lift` below measurement noise floor (`resources.md` §D4). Rationale strings cite sources not in `resources.md` — must be reconciled |
| `phases.yaml` | All thresholds tuned against inverted semantics. `landing_ankle_velocity_max` is `0.03` in YAML but the code default is `0.01` — silent divergence |
| `scoring.yaml` + `display.yaml` | **`summary_display_frames` defined in both.** `display.yaml` wins; `scoring.yaml`'s is dead |
| `filter_config.yaml` | Comment says "Tuned for 30 FPS" — assets are 24 fps |
| `pose.yaml` | Fine. `device: auto` correctly falls back to CPU on Windows |
| `hoop_roi.yaml`, `physics.yaml`, `ball.yaml`, `report_config.yaml` | **Not yet audited** — teammate-owned (ball/rim). Not touching without their input |

---

## 9. DEAD CODE / COMPLEXITY AUDIT

**Confirmed dead:**
- `JOINT_CHAINS["trunk"]` in [`angles/joint_chains.py`](../angles/joint_chains.py) references landmarks `mid_hip` / `mid_shoulder` that **do not exist** in `BASKETBALL_LANDMARKS`. `_compute_trunk_angle` never uses it. Misleading — **document or remove**.
- `scoring.yaml::summary_display_frames` — shadowed by `display.yaml`.

**Complexity hotspots** (flagged by static analysis): `pipeline.py::_process_ball` (w=24), `pipeline.py::__init__` (w=18). Both teammate-adjacent (ball/rim) — **do not refactor**.

**Preserved, not deleted (Part 37):** `ml/` pipeline, `physics/`, `filters/`, `ball/` visual tracking. All have entry points or teammate ownership.

---

## 10. MODEL AUDIT

`settings.py` loads `pose_landmarker_lite.task`; `README.md:7` claims **full**; all three variants are present in `models/`.

Google's documentation lists identical input dimensions and dtype for lite/full/heavy and **publishes no accuracy or latency comparison**. So a model change cannot be justified from documentation — it needs measurement.

**Recommendation:** do **not** switch blind. Benchmark lite vs full on the 7 videos post-Y-fix, measuring wrist/index landmark stability during release (the chain release detection depends on) and per-frame latency. Decide on evidence. Meanwhile **fix the README/settings contradiction** — one of them is lying to the next developer.

---

## 11. TEST STRATEGY

### 11.1 🚨 Root cause: the fixtures are the defect, not the code

[`tests/test_phases.py:17-25`](../tests/test_phases.py#L17-L25) builds `ankle_y=0.1, hip_y=0.6, shoulder_y=0.8, nose_y=0.9` — ankles low, head high, i.e. **Y-UP**. Real output is Y-DOWN. **The tests validate a world that does not exist**, which is why the bug survived. All three peer reviewers independently identified this.

The failure mode to avoid: fix the code, then "fix" the fixtures to match — and the tests become a check on self-consistency, not correctness.

### 11.2 The four layers

1. **Property/invariant smoke test (highest leverage).** Run real MediaPipe on a real frame and assert **ordering**, not values:
   `height_of(nose) > height_of(shoulder) > height_of(hip) > height_of(ankle)`
   This is what should have caught the bug, it makes the class of error structurally impossible to reintroduce, and it is a strong defence line.
2. **Real-landmark fixtures.** Harvest actual `pose_world_landmarks` from the 7 videos at known phase boundaries; freeze as `.npz`/JSON. These replace all hand-typed fixtures — converting unit tests into **characterisation tests against ground truth**.
3. **Golden-run regression.** Pin full-pipeline output (phase sequence, per-frame rule pass/fail, jump-signal sign) for all 7 videos. **The pre-fix baseline already exists** — the 0%/0%/66%-stuck numbers in §5.2 and §6.1. The pre/post diff is simultaneously the acceptance test and the thesis evidence.
4. **Height tests.** user-provided / missing / vision-unreliable / conflicting / unit-confusion / different heights.

### 11.3 The honest limitation
There is still **no coach-labelled ground truth**. "0% → X% pass" proves the rule is no longer always-failing; it does **not** prove X% is correct. That must be stated, not glossed.

---

## 12. HEIGHT ARCHITECTURE

```
PlayerProfile  (persistent, separate from frame data)
    height_cm: float | None          ← never fabricated
    height_source: user | vision | none
    height_confidence: high | float | None
    shooting_hand: auto | left | right
          ↓
   HeightProvider          ← thin resolver, not a framework
    ├── user input         (authoritative)
    └── vision estimate    (sanity check ONLY, never scoring)
          ↓
   Validated height (or None)
          ↓
   PlayerContext ──────────┐
                           ↓
MediaPipe → Y-UP boundary → filter → visibility → 3D angles
                           ↓
                        Features
                           ↓
              Shot classification (set / jump + confidence)
                           ↓
                     8-phase FSM
                           ↓
              Biomechanics evaluation
                           ↓
    Height-aware normalisation (ONLY release height)
                           ↓
                        Scoring
                           ↓
              Coaching feedback (external focus)
                           ↓
                      AR / EN
```

**Confidence definition (Part 4 — no invented numbers):**
- `user` → `high`. Not a probability; a provenance label.
- `vision` → a **defined** value or nothing. Until a validation study against known-height players exists, vision emits **no confidence number at all**.
- absent → `None`. **No default height is ever fabricated.**

**Council verdict:** player height is **scope creep on the bug-fix branch** and depends on `release_height` being correct first. **Defer to its own branch** (§13, Phase 4).

---

## 13. FULL ARCHITECTURE PLAN & SEQUENCING

The councils were unanimous that bundling these changes destroys reviewability and bisectability. **One concern per branch.**

| # | Branch | Content | Gate |
|---|---|---|---|
| **0** | `pose-edits` (current) | Research + `resources.md` + this plan | ✅ Done |
| **1** | `test/real-landmark-fixtures` | Harvest real landmarks; add ordering invariant test; pin golden-run baseline. **Tests written BEFORE the fix, proving the bug fails them.** | Baseline reproduces §5.2/§6.1 numbers |
| **2** | `fix/pose-y-axis` | Negate Y at `extract_world_landmarks`; fix `_resolve_shooting_side`; correct 3 docstrings; regenerate `ml/*.npz`; **migration note** | Golden-run diff shows trunk/release_height/jump recovered; skeleton overlays eyeballed on all 7 videos |
| **3** | `fix/phase-retune` | Retune `phases.yaml` against corrected data; fix the `knee_flexion` jam; reconcile the `landing_ankle_velocity_max` divergence | Phase distributions plausible vs video |
| **4** | `feat/timing-durations` | Frame counts → milliseconds; remove `DEFAULT_FPS` reliance | Identical behaviour at 24 and 30 fps |
| **5** | `chore/config-cleanup` | Dedupe `summary_display_frames`; fix README/model contradiction; reconcile rationale strings with `resources.md` | — |
| **6** | `feat/rule-evidence-alignment` | Demote `knee_flexion_loading` to metric; remove `shoulder_alignment_lift` from scoring; label engineering thresholds | Every rule traces to `resources.md` |
| **7** | `feat/player-height` | PlayerProfile, HeightProvider, release-height normalisation | Fairness tests across simulated heights |
| **8** | `feat/coaching-feedback` | External-focus rewrite; AR/EN; detection separated from localisation | Native-speaker review of Arabic |
| **9** | `chore/model-benchmark` | lite vs full measurement | Evidence-based decision |

### 🚨 Highest-risk step and its de-risking
**Step 2**, because a second masked axis error, or a downstream consumer that already half-compensated for the Y bug with its own hack, will only surface once Y is corrected. **De-risk:** run the golden-run diff *before* merging, and visually inspect rendered skeleton overlays on all 7 videos — not just metric thresholds.

### 🚨 Teammate protection (Parts 18, 44)
1. **`ml/*.npz` are a silent landmine.** They carry no schema version. After the negation they will load fine, train fine, and produce a confidently wrong model. → **regenerate from raw video** (never hand-patch) and add a version stamp.
2. **Ping the ball/rim owner directly.** `ball/release_sync.py` and `ball/shot_state_machine.py` consume `pose_phase` and wrist coordinates. They must be checked for Y assumptions — via a written note, not a diff they might miss.
3. **Written migration note**, not just a PR description: what is incompatible, who owns adjacent code, explicit "regenerate your `.npz`" instruction.

---

## 14. COUNCIL FINDINGS

Two councils were run (Parts 21 and 44), each with 5 independent advisors and an anonymised peer-review round. *Methodology note: 3 peer reviewers were used per council rather than 5, to conserve budget; reviews were highly convergent.*

### Council 1 — Research

**Agreement:** Claim 1 is a null-hypothesis fallacy, not evidence. Claim 2 is self-contradictory and must be cut from scoring. Claim 3's R²=0.005–0.012 must be stated explicitly. Claim 6 (monocular limits) is the *strongest* part of the thesis and should be argued from physics, not citations.

**Clash:** The Contrarian called the evidence base fatally weak; First Principles said the claims were never meant to bear that weight and should simply be **relabelled as tunable engineering priors**. → *Chairman: First Principles wins.* The fix is framing, not more citations.

**Blind spots caught in peer review:**
1. Nobody questioned **MediaPipe's own measurement error** — the same monocular limitation attacking a different variable. Now `resources.md` §D4.
2. **Motor-learning pedagogy was entirely absent** — and it contradicts our current feedback wording (§1.5).
3. **"Proficient compared to whom?"** — proficiency reference groups are mostly recreational adults and must be stated.
4. Nobody proposed **running our own small validation study** to stop relying entirely on borrowed evidence.
5. Height-entry **UX failure modes** unresearched.

**Strongest jury attack identified:** *"You built a technique-error classifier on the absence of statistical significance from one underpowered lab."*

### Council 2 — Engineering

**Agreement:** Option (a), negate at the boundary, is the structurally honest fix. **The fixtures are the real defect, not the code.** Retuning is mandatory, not a follow-up. Do not bundle unrelated changes. `.npz` files are a silent landmine.

**Clash:** Contrarian argued no fix can be *verified* without coach-labelled ground truth; Executor argued the existing broken-state numbers are a sufficient acceptance baseline. → *Chairman: Executor for sequencing, Contrarian for framing.* Use the golden-run diff as the gate, but state plainly that it proves "no longer always-failing", not "correct".

**Blind spot caught:** the suspiciously clean fit of true trunk lean (~6–14°) inside the existing `[5,22]` window is being read as confirmation the fix is right — but that window was hand-tuned by someone staring at broken 165–180° output. **It is not independent validation.**

### The one thing to do first
**Write the failing test before touching the fix.** Harvest real Y-DOWN landmarks from one video, assert the trunk rule passes on a known-good frame, and confirm it fails on current code. That single step converts every later claim from "we think we fixed it" into "here is the diff that proves it."

---

## OPEN DECISIONS FOR THE TEAM

1. **Confirm branch sequencing** (§13) — 9 branches, or compress for the deadline?
2. **Who owns `ball/`?** They need the migration note before step 2 merges.
3. **Can we record player heights** for even 3–5 people on the existing videos? That unblocks Parts 42–43.
4. **Arabic feedback** — native-speaker reviewer needed; the external-focus rewrite must not be a literal translation.
5. **Do we run our own validation study?** Council flagged this as the highest-value way to escape single-lab dependency. Even n=10 with a second synchronised camera would be defensible.
