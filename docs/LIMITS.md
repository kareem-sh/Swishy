# What this does not do

Written down so a limitation cannot quietly be presented as a result.

---

## Measured failures

| Case | What happens | Cause |
|---|---|---|
| Clip cut at the release | no shot found | no valley for prominence, no stance for elevation |
| Broadcast footage | 1 of 6 types correct | camera zoomed 4.7× in 2.5 s; a free throw measured 0.638 body heights of "rise" |
| `video9` | 8 test failures | stance baselines read negative; unresolved |
| A catch | scored as a shot | same body movement as a shot; needs the ball |
| Make / miss | mostly unanswered | ball outcome not joined to the offline path |

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

Jump shots and set shots / free throws. A driving action is reported as a layup
with low confidence and **never scored** — pose alone cannot separate a layup
from a hook shot from a dunk, and every member of that family is unimplemented,
so the outcome is the same whichever it was.

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
