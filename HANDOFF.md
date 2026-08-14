# Swichy — Session Handoff

**Read this first in a new session.** Written 13 Aug 2026, branch `pose-edits`.

Point a fresh session at this file and it has everything it needs to continue.

---

## What this project is

AI basketball **shooting coach for practice** (university thesis). Single camera
+ MediaPipe Pose (BlazePose GHUM), Python. Teaches technique. It is **not** game
analytics and **not** a make/miss predictor — `resources.md` §A6 is the evidence
for that scoping (R²=0.005–0.012 between kinematics and accuracy).

Scope is **Jump Shot + Set Shot only**. Everything else classifies to a
rejection, never to a score computed with the wrong biomechanics.

A friend is adding **ball + rim tracking** and will receive this repo.

---

## Hard rules (do not violate)

- No `if video_name == ...`, no `if fps == 12:`, no fixture-specific thresholds.
- **Everything is in SECONDS, never frame counts.** Footage runs 12–30 fps and
  includes slow motion.
- Player height is **user-entered only**. `height_cm = None` is a first-class
  state; never fabricate a default. Vision height must never feed scoring.
- Feedback keeps `[ON TARGET]` / `[REFINE]` / `[CHANGE]` and **external-focus**
  language (`resources.md` §C1).
- Never silently convert "not observed" into a value. The `or 0.0` bug — a
  rejected wrist landmark reading 0.0, which in hip-centred coordinates means
  "wrist exactly at hip height" — cost weeks. `wrist_world_valid` exists for this.

---

## Architecture as of now

### The phase model is TWO layers — this is the core design

Defined entirely in **`config/phase_model.yaml`**. Nothing is hardcoded in Python.

```
Layer 1 — detector (LIVE, 4 states)     phase_detection/detector.py
    ready → rise → release → recovery
    Small on purpose. Every state is a place a shot can get stuck and be LOST.
    Was 8 states: that meant 7 sequential threshold gates, and any one failing
    lost the whole shot. Measured: 4 of 10 shots on video8, 0 of 1 on video_07.

Layer 2 — analysis (POST-HOC, 7 phases)  feedback/phase_refiner.py
    ready_stance, loading, ball_lift, jump, release, follow_through, landing
    Assigned AFTER the shot is captured, by extrema over the window
    (argmin of knee angle = bottom of the dip). An argmin cannot get stuck;
    a live velocity threshold can, and did.
```

`knee_flexion` was **removed** — it carried the identical two rules as
`loading`, so one dip was reported twice under two headings.

### Segmentation is anchored on the RELEASE

`feedback/shot_tracker.py`. When the detector reports `release` and no candidate
is open, the shot window is **rebuilt backwards** out of `FrameBuffer`
(300 frames = 10 s).

Why: attempts could only open from rest, so recovery (3–5 s) had to finish
before the next attempt could start — but the gap between attempts in practice
footage is ~2.4 s. Every recovery swallowed the next shot. Four separate
retunings each fixed one fixture and broke another. It was a sequence that
could not fit, not a threshold that needed tuning.

Every candidate also **back-fills** ~1.5 s of pre-roll, so the knee dip is
analysed even though nothing in the detector looks for it.

---

## Current measured state

| Fixture | Truth | Before | After |
|---|---|---|---|
| `video8.mov` | 10 | 4 | **9** (one wrongly typed `layup`) |
| `video9.mov` | 3 | 2 | **3** ✅ |
| `video_07` | 1 | 0 | **1** (type wrong: `set`, should be `jump`) |
| `video_01` | 2 | 2 | **4** ⚠️ regression |
| `video_04` | ? | 1 | **0** ⚠️ uninvestigated |
| 13 single-shot clips | 13 | 13/13 | **13/13** |

`136 passed, 5 xfailed, 0 failed`

### Numbers that flatter — state these honestly

1. "9 of 10" is really **8 correct + 1 wrong type + 1 missed**.
2. `video_01` 2→4 is a **real regression**. Verified from footage: 2 shots only.
   The extras are (a) the recording opening mid-follow-through of an earlier
   shot, and (b) the player **catching** the returning ball at 6.6–9.1 s.
3. `video_07` 0→1 is half a win — it now answers, and answers wrong. The
   project's own test philosophy says declining is honest and a confident wrong
   answer is not.
4. **13/13 is not independent evidence** — we cut those clips ourselves at
   midpoints, so each contains one shot by construction.
5. **Ground truth itself is unresolved**: the user says the last TWO shots in
   video8 are jumps, but `video8_shot09` is labelled `set` and was never
   visually verified.
6. The 5 `xfail` are real documented failures, `strict=True`.

---

## THE NEXT THING TO DO (agreed direction, not yet started)

The user asked for the *right* method rather than more patching, and accepted
this analysis. **Do not just keep tuning the FSM.**

### The real problem, in three layers

1. **Definitional** — a shot is defined by the BALL; we measure the body. Every
   failure traces here. Catch vs shoot is the same pose.
2. **Architectural — the big one.** We built a *streaming* detector
   (`process_frame` one frame at a time, no lookahead) for an *offline* task
   (user uploads a video, gets a report). Timeouts, hysteresis, back-fill and
   the `_went_up` latch all exist only to compensate for not seeing the future.
   The whole video is available from frame zero.
3. **Methodological** — an FSM commits to a state and gets stuck. Every
   "sink state" bug was structural to that choice.

### Recommended replacement, in priority order

**1. Replace the FSM with offline peak detection.** Highest value by far.

```python
from scipy.signal import find_peaks, peak_widths
peaks, _ = find_peaks(wrist_height_ratio,
                      prominence=0.25,      # body heights
                      distance=fps * 1.0)   # min gap between shots
left, right = peak_widths(wrist_height_ratio, peaks, rel_height=0.9)[2:]
```

`prominence` is a rigorous version of exactly what `_track_climb`'s running
floor hand-rolls today. This deletes: timeouts, hysteresis, back-fill, the
latch, `_consumed_until_ms`, and the moving-floor logic. **`phase_refiner.py`
survives unchanged** — it is already built on this principle, which is why it
never caused trouble.

**2. Global ankle baseline.** One line, fixes `video_07`:
```python
ankle_baseline = np.median(ankle_y[player_is_still])   # whole video
```
Today it adapts per FRAME and creeps upward during flight — worst on the only
slow-motion fixture. This is the last per-frame constant in a per-second codebase.

**3. Ball as VERIFIER, not detector.** Counter-intuitive but important: ball
tracking in real practice footage is unreliable (small, motion-blurred,
occluded). If it drives detection, every tracking failure = a lost shot.
Instead: peak detection proposes candidates → ask the ball only at each peak
"did it separate from the wrist and move away?" → that rejects the catch. Clean
division of labour with the friend: hand him a short list of timestamps, not a
whole video to track.

**4. Learned model = future work.** Temporal Action Segmentation (TCN/BiLSTM)
is the academically correct answer, but there are only ~15 labelled shots. It
would memorise the fixtures. For the thesis, state that the signal-processing
method was chosen deliberately given the data size — that is a strong defence,
not an apology.

---

## Known limitations, each with an owner

| Limitation | Owner |
|---|---|
| Catch is not separable from a shot by pose alone | **The ball** (friend's work) |
| `body_rise_ratio` reads 0.031 on `video_07` vs 0.50 on video8/9 | Global ankle baseline (item 2) |
| `video_04` 1→0 | Uninvestigated |
| video8 shot #9: is it a jump? | Needs visual verification |

**Correction to an old assumption in the code:** it was written that `video_07`
fails because the player is distant / low resolution. Measured and false — the
player fills **0.463** of frame height vs **0.212** in video8. Resolution was
never the problem.

---

## Things tried and REJECTED on measurement (do not retry)

All recorded in code comments with their reasons.

| Idea | Result |
|---|---|
| Require elbow extension to confirm release (to reject catches) | Reaching for a catch extends the arm too. Didn't separate them, and cost a real shot in video8 (9→8) |
| Widen the elevation window into `recovery` (apex is after release) | Tried unbounded and bounded to 0.8 s flight. No gain on the target case; turned a set shot into a jump |
| `post_release_s` 1.2 → 2.5 for slow motion | Zero change in every number |
| Absolute timeout cap on `ball_lift` (old 8-state FSM) | video8 4→5 but video_01 2→1 |
| `refractory_s` 0.60 → 0.25 | Gained one shot, lost another |

---

## Test suite

| File | What it guards |
|---|---|
| `tests/test_phase_integration.py` (9, new) | Whole chain, nothing mocked, on a long multi-shot video **and** a single clip. Key guard: **every rule in `biomechanics.yaml` targets a phase the refiner can emit** — a rule pointing at a deleted phase never fires, the shot still scores, nothing raises, and one coaching section is silently missing |
| `tests/test_phase_refiner.py` (7, new) | Deepest knee bend lands inside `loading`; timeline never runs backwards; a set shot correctly has NO `jump` phase |
| `tests/test_phases.py` (11) | Detector contract + `len(CORE_STATES) <= 4` structural guard |
| `tests/test_acceptance_videos.py` (53) | Real videos through the real pipeline |

Run: `./venv/Scripts/python.exe -m pytest tests/ -q -p no:warnings`
(the venv matters — system python has no `cv2`)

---

## Research corpus

- **`resources.md`** — authoritative evidence map. Read this for what each paper
  supports and what it must NOT be used for.
- **`PAPER_LINKS.md`** — resolvable link for every source, with two status axes:
  is the link verified, and how much of the paper was actually read (only
  **4 of ~20 were read in full**; most are abstract-only).
- **Two live citation problems** are documented in `PAPER_LINKS.md` and NOT yet
  fixed: `docs/BIOMECHANICS_RESEARCH.md` misattributes A1 to the wrong
  journal/authors, and still cites a source `resources.md` excludes as
  predatory-adjacent.

---

## Environment

- Windows, PowerShell. Use `./venv/Scripts/python.exe`.
- Videos in `assets/videos/`; generated clips in `assets/videos/single_shot/`
  (gitignored — rebuild from `manifest.json`).
- Full-suite acceptance runs take ~2–5 minutes.
