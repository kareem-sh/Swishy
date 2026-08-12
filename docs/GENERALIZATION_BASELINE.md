# Generalization Baseline — does the pipeline work beyond its two fixtures?

**Branch:** `pose-edits` · **Pipeline state:** unchanged (Y-UP fix `9269000` only; **no threshold was touched**)
**Reproduce:** `python scripts/generalization_baseline.py <video> [...]`

All durations are reported in **seconds**, never frame percentages, because the
material spans 12–30 fps and one clip is slow-motion.

---

## Scope — decided, not assumed

> **Swichy is a free-throw / set-shot / jump-shot analyser.** (Option A)

Layups, hook shots, dunks, tip-ins and one-legged shots have different mechanics
and are **out of scope**. The jump-shot rules will **not** be distorted to
accommodate them.

This has a direct consequence that the baseline exposes: an out-of-scope shot
must be **refused**, not silently scored. Today it is silently scored (§4.6).

---

## 1. Videos tested

| Video | Action type | In scope | Source | FPS | Duration |
|---|---|---|---|---|---|
| `video_01_free_throw` | Free throw / jump shot | ✅ | fixture | 24.00 | 13.50 s |
| `video_02_one_on_one` | Contested shots, **defender present** | ⚠️ game context | fixture | 23.98 | 8.01 s |
| `video_06_dunk` | Dunk | ❌ out of scope | fixture | 24.00 | 10.33 s |
| `video_07_side_jump_shot` | Jump shot, **slow motion** | ✅ | fixture | 29.99 | 45.05 s |
| `external_01_free_throw_curry_kerr` | Free throw | ✅ | YouTube CC-BY | 29.97 | 14.75 s |
| `external_02_jump_shot_peja` | Jump shot, repeated reps | ✅ | YouTube CC-BY | **12.00** | 50.25 s |
| `negative_control_layup_van_rossom` | Layup | ❌ negative control | Wikimedia CC BY-SA | **1000** (bogus) | 9.10 s |

Provenance and licences: [`assets/videos/external/README.md`](../assets/videos/external/README.md).

---

## 2. Baseline results

| Video | Expected shots | Detected | Result |
|---|---:|---:|---|
| `video_01_free_throw` | 2 | **2** | ✅ count correct — but see §4.2 |
| `video_02_one_on_one` | 2 | **2** | ✅ count correct — scores 3 and 31 |
| `video_06_dunk` | 2 (out of scope) | **2** | ❌ scored an out-of-scope action |
| `video_07_side_jump_shot` | ~1 | **7** | ❌ severe over-segmentation |
| `external_01_free_throw` | 2–3 | **4** | ❌ over-segmentation |
| `external_02_jump_shot` | ~4 | **6** | ❌ over-segmentation |
| `negative_control_layup` | 0 (should refuse) | **1**, score 40 | ❌ silently scored |

### Phase time, seconds

| Video | ready | loading | knee_flex | ball_lift | jump | release | follow | landing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `video_01` | 4.67 | 0.17 | — | 0.38 | **5.21** | 1.58 | 0.54 | 0.96 |
| `video_02` | 0.63 | 0.29 | 0.17 | 0.38 | 0.38 | 0.33 | **4.21** | 1.63 |
| `video_06` (dunk) | 7.38 | 0.88 | — | 0.17 | **0.00** | 0.83 | 0.17 | 0.92 |
| `video_07` (slow-mo) | 19.81 | 3.30 | 6.47 | 4.03 | 0.73 | **9.70** | 0.60 | 0.40 |
| `external_01` | 5.67 | 0.67 | — | 0.60 | **0.00** | 1.30 | 2.34 | **4.17** |
| `external_02` | 15.17 | 3.00 | 6.25 | 0.83 | **10.08** | 4.50 | **9.25** | 1.17 |

---

## 3. What currently works

1. **Pose detection generalises well.** MediaPipe found a pose on essentially every frame across all resolutions tested (320×240 to 1280×720). Not a bottleneck.
2. **The set-shot path is correct and is being used.** Free throws (`external_01`) and the dunk (`video_06`) both reached `release` **without ever entering `jump`** (0.00 s). The FSM's non-jump branch works, and jump-shot mechanics are **not** being force-fitted onto shots that lack a jump.
3. **Multi-shot segmentation exists and does not leak.** `video_02` produced two independent shots with separate phase sequences and separate scores (3 and 31). No state carried across. The requirement in the brief is already met structurally.
4. **Every video reached `release` and `follow_through`.** The pipeline never fails outright; it always produces something.
5. **The Y-UP fix holds on unseen footage** — no reversion on any external clip.

---

## 4. What fails

### 4.1 Over-segmentation is the dominant failure
Five of seven videos detect more shots than exist. Worst: `video_07` reports **7** for roughly one shooting sequence.

**Root cause is in `ShotTracker`, not `phases.yaml`.** `_is_start_transition` opens a new shot on *any* `ready_stance → loading|ball_lift` transition. There is no minimum shot duration, no requirement that a shot contain a release, and no refractory period. Any re-settle, toe-bounce or aiming adjustment starts a new "shot".

> **This is why threshold tuning cannot fix it.** Lowering or raising a threshold in `phases.yaml` changes *when* transitions fire, not *whether a burst of transitions counts as one attempt*.

### 4.2 `jump` absorbs the aiming period
`video_01` shot 1: `jump@0.33s → release@3.29s`. That is **2.96 seconds of "jump"**. Human hang time is ~0.5–0.9 s. The FSM enters `jump` during the lift and cannot leave until release conditions are met, so `jump` becomes a waiting room. Same pattern in `external_02` (10.08 s total).

### 4.3 `follow_through` and `landing` behave as sink states
`video_02` spends **4.21 s** (53% of clip) in `follow_through`; `external_01` spends **4.17 s** in `landing`. Their exits require signals that may never arrive once the player walks away or leaves frame. Nothing forces a timeout.

### 4.4 Frame-count thresholds do not survive the frame-rate range
`hysteresis_frames: 2` means 67 ms at 30 fps but **167 ms at 12 fps** (`external_02`). `VISIBILITY_HOLD_FRAMES = 5` means 167 ms vs 417 ms. Confirms the timing audit — these must become durations.

### 4.5 No frame-rate sanity check
The layup container reports **1000 fps**. The pipeline accepted it, computed nonsense timestamps (every phase 0.00 s), and **emitted no warning**. Any implausible fps should be rejected or clamped with a visible warning.

### 4.6 🚨 Out-of-scope shots are silently scored
The layup received **1 detected shot, score 40/100**. The dunk received **two shots, scores 35 and 60**. Under Option A scope this is a correctness defect: the system asserts a jump-shot judgement over an action whose mechanics it does not model. A young player would be told to fix their "release height" on a layup.

### 4.7 Scores are not yet trustworthy
`video_02` scored **3/100**. That is a contested game shot with a defender — outside the practice framing — but a 3 indicates the rules are firing on frames that are not really the labelled phase, downstream of §4.1–4.3.

---

## 5. Answers to the diagnostic questions

**1. What works?** Pose detection, the non-jump set-shot path, multi-shot independence, and the Y-UP fix. See §3.

**2. What fails?** Shot segmentation (§4.1), phase dwell (§4.2–4.3), frame-rate handling (§4.4–4.5), and the absence of a scope gate (§4.6).

**3. Segmentation or phase detection?**
**Primarily segmentation.** Phase *ordering* is correct everywhere — every video walks a sensible `ready → loading → lift → (jump) → release → follow_through → landing`. What is wrong is (a) how long phases persist, and (b) how transition bursts are grouped into attempts. Grouping lives in `ShotTracker`; dwell lives in `phases.yaml`. Both need work, and **segmentation must be fixed first**, because dwell measurements are meaningless while attempt boundaries are wrong.

**4. Are jump-shot assumptions leaking into non-jump shots?**
**No — and this is the strongest positive result.** The free throw and the dunk both reached release with `jump` at exactly **0.00 s**. The FSM does not require a vertical jump. The leak is in the opposite direction from what was feared: the problem is not that jump mechanics are forced onto set shots, but that **no shot type is rejected at all** (§4.6).

**5. Does `config/phases.yaml` need retuning?**
**Yes, but not first, and not alone.** Retuning cannot fix over-segmentation (§4.1) or the sink states (§4.3), both of which are structural. Retuning against the current attempt boundaries would fit thresholds to mis-segmented data — repeating the original mistake of tuning against a broken substrate. Order must be: segmentation → dwell/timeouts → threshold values.

**6. Does the system need shot-type classification before phase analysis?**
**Not full classification — but it does need a scope gate.** Since scope is Option A, Swichy does not need to *analyse* layups; it needs to *decline* them. A cheap, evidence-backed discriminator already exists: **vertical displacement** (~15 cm free throw vs ~27–31 cm jump shot, `resources.md` A1), plus whether the ball is released above or below head height. That is enough to answer "is this a free throw, a set shot, a jump shot, or something we do not support?" A full layup/hook/dunk classifier is unnecessary and out of scope.

---

## 6. Recommended order of work (evidence-driven, no overfitting)

1. **Shot-segmentation gate** in `ShotTracker` — minimum attempt duration, require a `release` phase for an attempt to count, and a refractory period after landing. Directly addresses §4.1 and the toe-jiggling requirement.
2. **Phase timeouts** — cap `follow_through` and `landing` dwell so they cannot become sinks (§4.3).
3. **Frame-rate validation** — reject or clamp implausible fps with a visible warning (§4.5).
4. **Frame counts → durations** in `phases.yaml` and `settings.py` (§4.4).
5. **Scope gate** — classify free throw / set shot / jump shot from vertical displacement and release height; anything else returns "unsupported shot type", not a score (§4.6).
6. **Only then** retune threshold *values* against corrected attempt boundaries.

### Constraints on that work
- No `if video_name == ...`, no per-video thresholds.
- No lowering a threshold purely to make a chosen clip pass.
- Re-run this baseline after each step and diff against the table in §2.

---

## 7. Honest limitations of this baseline

- **Expected shot counts are my visual estimate**, from frame grids, not an independent labelling. `video_07` in particular is 45 s of slow motion and may genuinely contain more than one attempt; "7" is still implausible, but "1" is not certain either.
- **`video_07` is slow-motion**, so its wall-clock seconds are not physically comparable to the other clips. Its *relative* phase proportions and its shot count are the meaningful signals, not absolute durations.
- **Only 2 in-scope external videos** were added, per the brief. Two clips cannot establish generalisation; they can only demonstrate failure, which they did.
- **No coach-labelled ground truth** for phase boundaries. "Implausible" durations here are judged against human movement physics (hang time, follow-through length), not against labelled data.
