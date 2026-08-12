# Swichy — Acceptance Sheet: World-Coordinate Y-Axis Fix

**Scope:** the fix that made vertical measurements (trunk lean, release height, jump detection, phase transitions) mean what the code assumed they meant.
**Audience:** anyone verifying the fix without reading test code.
**Written against:** commit `9269000` + the `tests/test_coordinates.py` suite.

---

## 1. What this checks, in plain language

MediaPipe reports every body joint as a 3D point, and each point has a vertical coordinate, "Y". The whole Swichy codebase was written assuming **larger Y means physically higher** — head has a big Y, feet a small Y. That assumption was wrong for the data stream Swichy reads (`pose_world_landmarks`): MediaPipe reports it the other way round, head at a *smaller* Y than the feet.

Think of a thermostat wired backwards: turning the dial toward "warmer" makes the room colder. Swichy's vertical axis was wired backwards — and nobody noticed, because the hand-written test fixtures were wired backwards in exactly the same way. **The tests were checking the code against itself, not against reality.**

This sheet checks two separate things:
1. the axis is now the right way round, and
2. quantities that never depended on "which way is up" — such as the elbow angle — were never broken and still are not.

---

## 2. How to run it

```bash
# from the repository root, using the project virtual environment
venv/Scripts/python.exe -m pytest tests/ -q
```
**Expected:** `65 passed`, no `FAILED` lines.

```bash
# regenerate the real-landmark fixtures from the live model
venv/Scripts/python.exe scripts/build_test_fixtures.py
```
**Expected:** `wrote 4 samples -> tests\fixtures\real_world_landmarks.json`, with no `!!` lines.

> The pass-rate percentages in §4 come from running the analysis pipeline over whole videos and counting rule outcomes per frame. That is a separate, heavier measurement than `pytest` — see §7.4.

---

## 3. Use cases

| # | Use case (plain language) | What it proves | Expected result | Automated? |
|---|---|---|---|---|
| 1 | **Head above feet** — in the numbers the model outputs, is the nose really above the shoulders, hips, knees and ankles? | The axis direction is correct at the single point where MediaPipe data enters the system, checked against **real recorded** landmarks. | `nose.y > shoulder.y > hip.y > knee.y > ankle.y` on every fixture sample. | ✅ `test_anatomical_vertical_ordering` |
| 2 | **The axis constant is exactly a Y flip** and applying it twice returns the original. | The fix is a reflection, not some other transform — so a stray double-flip shows up as "the bug came back", not a new failure mode. | `MEDIAPIPE_TO_SWICHY == [1, -1, 1]`. | ✅ `test_mediapipe_to_swichy_is_exact_y_negation` |
| 3 | **Upright person measures as upright.** | `angle_from_vertical` compares a segment against a fixed `up = [0,1,0]`. Only correct if input is already Y-up. | Real hip→shoulder segment reads **0–45°**, not 165–180°. | ✅ `test_real_trunk_segment_lean_is_plausible` |
| 4 | **Trunk angle through the real production path.** | Landmarks → VisibilityGate → AngleCalculator yields a valid, plausible trunk lean. This is the rule that used to pass 0 of 1359 frames. | Valid result, 0–45°. | ✅ `test_trunk_angle_via_angle_calculator_is_plausible` |
| 5 | **Joint angles unaffected by a frame flip.** | The dot-product formula is invariant under reflection — this is *why* elbow/knee/hip/shoulder survived the bug untouched. | Angles identical to within 1e-6 after negating every coordinate. | ✅ `test_joint_angles_are_reflection_invariant_on_real_data` |
| 6 | **Rising = positive velocity.** | The jump detector fires on `ankle_rise > 0.03`. Pre-fix, a real jump produced **−0.118** — right magnitude, wrong sign — so it could never fire. | Wrist moving up → `wrist_velocity_y > 0`; down → `< 0`. | ✅ `test_wrist_velocity_positive/negative_...` |
| 7 | **Screen coordinates deliberately left alone.** | Only `extract_world_landmarks` is flipped. Image landmarks must stay in screen convention or the skeleton overlay renders upside down. | Image-space nose `y` **smaller** than ankle `y`, while world-space nose `y` is **larger**. | ✅ `test_image_landmarks_are_not_flipped` |
| 8 | **End-to-end rule recovery on real footage.** | `trunk_posture` and `release_height` were structurally incapable of passing, regardless of shooting quality. | Pass rate rises from 0% to a non-zero, video-dependent value (§4). | ⚠️ Manual — see §7.4 |

### Proof that these tests actually catch the bug
The axis constant was temporarily reverted to `[1, 1, 1]` (reproducing the original defect) and the suite re-run: **14 tests failed**, including anatomical ordering, trunk plausibility, and the production-path trunk angle. Restoring the constant returned all to green. The tests are therefore a real gate, not a restatement of the code.

---

## 4. Before/after evidence

Measured on real sample videos — 1821 frames before, 1675 after (different frame counts because `has_pose` differs once phase behaviour changes).

| Metric | Before | After | Rule window |
|---|---|---|---|
| Y ordering (head above feet) | Y-DOWN **1821/1821** | Y-UP **1675/1675** | structural |
| Trunk lean, video 1 (free throw) | 166–180° | **0.2–14.0°**, mean 5.6° | `[5, 22]°` |
| Trunk lean, video 7 (side jump shot) | 165–177° | **3.0–14.6°**, mean 10.0° | `[5, 22]°` |
| `trunk_posture` pass rate, video 1 | **0%** (0/107) | **60.4%** | inside `[5,22]°` |
| `trunk_posture` pass rate, video 7 | **0%** (0/1211) | **89.7%** | inside `[5,22]°` |
| `release_height` pass rate, video 1 | **0%** | **100%** | inside window |
| `release_height` pass rate, video 7 | **0%** | **0%** — different cause, see §7.3 | inside window |
| Jump signal on a real jump | **−0.118** (wrong sign) | **+0.118** | fires `> +0.03` |
| Phase FSM, free throw | stuck in `knee_flexion` **213/324 (66%)** | no longer jams | — |
| Unit tests | 42 passed (while product broken) | **65 passed** | — |

---

## 5. How to tell if you PASSED

- [ ] `venv/Scripts/python.exe -m pytest tests/ -q` → **65 passed**, zero `FAILED`
- [ ] `venv/Scripts/python.exe scripts/build_test_fixtures.py` → `wrote 4 samples`, no `!!` lines
- [ ] `pose/landmarks.py::MEDIAPIPE_TO_SWICHY` is `[1.0, -1.0, 1.0]`, applied **only** inside `extract_world_landmarks`
- [ ] `geometry/vectors.py::angle_from_vertical` uses `up = [0.0, 1.0, 0.0]`
- [ ] On a video with a standing shooter, trunk lean reads single digits to low twenties — not 150°+
- [ ] Skeleton overlay still renders right-side-up
- [ ] `ankle_rise` is **positive** during a real jump

All boxes ticked ⇒ the axis fix is in place and internally consistent. It does **not** follow that the biomechanics rules are correctly *tuned* — see §7.

---

## 6. How to tell if something REGRESSED

| Symptom | Likely cause |
|---|---|
| Trunk lean reads 150–180° again | The flip in `extract_world_landmarks` was removed or bypassed by code reading `pose_world_landmarks` directly |
| Flip looks applied but nothing changed | Applied **twice** — grep for a second `* -1` or `MEDIAPIPE_TO_SWICHY` outside `pose/landmarks.py` |
| Elbow/knee/hip/shoulder values change after axis work | Something outside `angle_between_vectors` now depends on absolute Y sign — out of scope by construction |
| Skeleton overlay upside down | The world flip leaked into `extract_image_landmarks` |
| Jump signal negative during a real jump | Sign convention regressed — same root cause as the original bug |
| FSM stuck in `knee_flexion` for most of a shot | The jam condition (`wrist_y > hip_y_avg`) is Y-sign dependent; the axis flipped back |
| `pytest` stays green through any of the above | **Should now be impossible** — `tests/test_coordinates.py` is driven by real recorded landmarks. If it happens, the fixtures were hand-edited instead of regenerated |

---

## 7. Known limitations — what these tests do NOT prove

State these plainly to a jury; overstating is the fastest way to lose credibility.

**7.1 No coach-labelled ground truth exists.** The jump from 0% to 60.4%/89.7% proves the trunk rule **stopped always-failing**. It does **not** prove 60.4% is the correct proportion of well-executed shots. "The rule can now pass" and "the rule is calibrated correctly" are different claims; only the first is supported here.

**7.2 Phase thresholds are known to be mistuned.** They were hand-tuned while the bug was present, i.e. against inverted semantics. After the fix, `video_01` spends 125/324 frames (39%) in `jump` and `video_07` spends 291/1351 (22%) in `release` — both implausibly high. A retune is the next task, not something this fix delivered.

**7.3 `release_height` at 0% on video 7 is a downstream symptom, not a metric defect.** It is caused by `release` over-triggering (§7.2), so the rule is evaluated on frames that are not the release moment. The metric itself (`wrist_y − nose_y`) is not implicated.

**7.4 The §4 percentages are not reproduced by `pytest`.** They came from running the full pipeline over whole videos and aggregating rule outcomes per frame. A green unit suite is necessary but not sufficient evidence that those percentages still hold.

**7.5 The older suites still use hand-typed fixtures.** `tests/test_angles.py` and `tests/test_phases.py` use synthetic values that now happen to agree with the corrected code. They cannot, alone, catch a future sign error. `tests/test_coordinates.py` is the actual defence, because it reads real recorded model output.

**7.6 `knee_flexion_loading` is out of scope for this fix.** Independent of the axis bug, its research support was found contradictory (three directions across four sources), and it is being demoted from a scored rule to a displayed metric. Do not read its pass rate as evidence about this fix.

**7.7 Monocular scale ambiguity is unaffected.** A single camera cannot resolve absolute scale without extra information. This fix corrects axis *direction*, not the system's inability to measure true distances — which is why player height is user-entered, not vision-estimated.
