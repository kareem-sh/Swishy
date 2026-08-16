# What this does not do

Written down so a limitation cannot quietly be presented as a result.

---

## Measured failures

| Case | What happens | Cause |
|---|---|---|
| Clip cut at the release | no shot found | no valley for prominence, no stance for elevation |
| Broadcast footage | 1 of 6 types correct | camera zoomed 4.7× in 2.5 s; a free throw measured 0.638 body heights of "rise" |
| `video9` | 8 test failures | stance baselines read negative; unresolved |
| `video8.mov` shot 4 | typed jump, is a set shot | borderline elevation; flipped when the One Euro lag was fixed. Same clip in isolation types correctly. No test covers it |
| A catch | scored as a shot | same body movement as a shot; needs the ball |
| Make / miss | mostly unanswered | ball outcome not joined to the offline path |

---

## How precise an angle is

Measured, not estimated. There is no protractor in the footage, so accuracy
cannot be checked directly — but uncertainty can be **bounded** without any
ground truth, by running the same frames through MediaPipe's own lite, full and
heavy models. All three are Google's and equally plausible; wherever two differ,
one is wrong by at least that much. Reproduce with
`scripts/measure_angle_uncertainty.py`.

`full` vs `heavy`, 448 sampled frames of `salah_video.mp4`:

| Angle | mean | median | p90 | disagree > 10° |
|---|---|---|---|---|
| **Elbow** | **11.5°** | 9.2° | 23.8° | **46%** |
| Knee | 8.9° | 7.3° | 17.2° | 32% |
| Shoulder | 8.6° | 5.7° | 17.8° | 26% |
| Hip | 6.2° | 4.6° | 12.0° | 16% |

Two findings matter more than the table itself.

**Confidence does not identify the reliable frames.** Restricted to frames
where every landmark scores `visibility > 0.9`, elbow disagreement *rose*
slightly (11.5° → 11.6°) and the share differing by more than 10° went from 46%
to 52%. You cannot filter your way to precision using MediaPipe's own score.

**`heavy` is not the fix.** Its forearm length varies more than `full`'s
(CV 18.1% vs 14.8%). The larger model reconstructs the arm *worse* here.
Bone-length variation overall runs 5–18%: a forearm was reconstructed as short
as 4 cm on some frames, which is anatomically impossible and marks frames where
the depth estimate collapses.

### What this means for the rules

Band widths in `config/biomechanics.yaml` must be read against these numbers.

| Rule | Outer band | Ideal band | Uncertainty | Verdict |
|---|---|---|---|---|
| Elbow at release | 142–180 (**38°**) | 152–168 (**16°**) | ~11.5° | outer sound, ideal marginal |
| Knee at loading | 80–145 (65°) | 100–128 (28°) | ~8.9° | sound |
| Hip at loading | 110–168 (58°) | 125–152 (27°) | ~6.2° | sound |

So `[CHANGE]` — a value outside the outer band — is trustworthy: a deviation
that large is real. But the split between `[ON TARGET]` and `[REFINE]` on the
**elbow** is a weak signal, because the ideal band's half-width (8°) is about
the size of the measurement uncertainty. It is reported, and deliberately not
narrowed. Narrowing it would borrow a precision that does not exist.

The defensible claim is *"this system detects gross deviations from a shooting
pattern"*. The indefensible one is *"your elbow was 157°, target 160°"* — a 3°
claim on an 11° instrument.

---

## Requirements, not bugs

**~0.8 s of the player standing before the shot.** Measured: 2/5 detected at
0.5 s of lead-in, 5/5 at 0.8 s, no further gain beyond that. Both the
segmentation and the shot-type measurement need those frames, for different
reasons.

**A roughly still camera.** Every measurement is expressed relative to
something in the image — the floor under the player, the player's own height.
A camera that pans or zooms moves the references. `ball/ideal_trajectory.py`
already solves this on the ball side by working in rim-radius units; the pose
side does not, and that is the honest next step for handheld footage.

**One player, whole body in frame.** MediaPipe tracks one person, and nothing
guarantees it picks the shooter.

---

## Scope

Jump shots and set shots / free throws. Nothing else.

A driving action is reported as **unrecognised** and **never scored**. Not as a
layup: the horizontal test measures that the attempt is not stationary
shooting, and says nothing about what it is instead — pose alone cannot
separate a layup from a hook shot from a dunk. Naming the most common member
of a family we cannot resolve would be a guess wearing a label.

That refusal is a **guard, not a feature**. Removing it would not narrow the
product to jump and set; it would send drives into the jump-shot rules and
produce confident, wrong coaching.

---

## Evidence limits

Stated plainly because a reviewer who finds these before we disclose them is
the worst outcome.

- **Single-lab dependency.** Most numeric shooting-kinematics data comes from
  one research group, in ESCI-only journals, n=10–34, often recreational
  participants. There is no independent replication.
- **4 of ~20 papers were read in full.** The rest are abstract-only, marked as
  such in `SOURCES.md`. Abstract-only numbers are never presented as verified.
- **Monocular scale ambiguity is fundamental**, not an implementation gap. A
  single camera cannot separate a tall player from a near one. Player height is
  user-entered only; vision height never feeds scoring.
- **No coach-labelled ground truth** for phase boundaries. "Plausible" here is
  judged against human movement physics, not against labelled data.
- **`0.12`, `0.55`, `1.5 s`** and the rest are engineering decisions calibrated
  against this project's footage. They are labelled as such in the code and are
  not published norms.

---

## Known-broken tests

19 failures against a baseline of 5. The 5 are pre-existing rim-crossing
semantics in `ball/`. Of the 14 introduced by the move to offline
segmentation, 8 are `video9`.

They are left failing rather than deleted or relaxed. A test that fails for a
documented reason is information; a test quietly adjusted until it passes is
the opposite.

---

## Deliberately not built

**A learned temporal model** (TCN / BiLSTM per-frame phase segmentation) is the
academically correct answer and was not built. There are roughly 15 labelled
shots and no frame-level labels at all; a model trained on that would memorise
these clips and fail unexplainably, which is worse than a threshold that fails
legibly. The signal-processing route was chosen knowing this — a position to
defend, not to apologise for.

A realistic middle step exists: a small classifier at each detected peak
answering *shot or catch*, which needs ~12 labelled peaks per video instead of
900 labelled frames, and turns today's false positives into training data.
