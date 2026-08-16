# temporal/ — the sequence model

Everything for the temporal AI model lives here: dataset preparation, the
model, training, and the accuracy harness. Nothing here is imported by the
production analysis path, and nothing here may change it.

Named `temporal/` and not `training/` on purpose: `assets/videos/training/` is
a **video directory**, and two things called "training" one holding data and
one holding code is a mistake waiting to happen.

## Layout

```
temporal/
  dataset.py               inventory -> labels -> groups -> split -> manifest.json
  preprocess.py            crop + upscale the clips the detector struggles with
  extract_shots.py         manifest -> shots.json, the actual training index
  compare_preprocessing.py A/B the preprocessing stage against raw footage
  evaluate.py              score a model's predictions without flattering them
  reference/               code recovered from the deleted ml/, read-only
  data/                    manifests and shot records; clips/ is gitignored
  checkpoints/             trained weights                 (gitignored)
  artifacts/               evaluation output               (gitignored)
```

## Order of operations

```
venv/Scripts/python.exe temporal/dataset.py          # manifest + split
venv/Scripts/python.exe temporal/preprocess.py       # render data/clips/
venv/Scripts/python.exe temporal/extract_shots.py    # data/shots.json
```

`dataset.py` rebuilds the manifest from scratch each run and carries
`prepared_path` forward from the previous one, so re-running it does not
silently send extraction back to the raw footage.

To re-check whether preprocessing still earns its place:

```
venv/Scripts/python.exe temporal/extract_shots.py --raw
venv/Scripts/python.exe temporal/compare_preprocessing.py
```

## The unit of this dataset is a SHOT, not a clip

`salah_video.mp4` is one file and five shots. A manifest keyed by filename
cannot be the training index, which is why `shots.json` exists and why every
key is `<clip>#<shot_number>`.

And the count that matters is not the clip count. **Under half of the detected
shots carry a usable elevation at all** — the rest are missing for stated
reasons recorded per shot in `no_elevation_reason`, never filled in with a
zero. Read that number before deciding what size of model is justified.

Weights, logs and feature arrays are **never committed**. They are rebuilt from
the videos and the pipeline, and a checkpoint in git goes stale the moment a
band or a phase cut changes with nothing to warn you that it has.

## `reference/` — read it, do not import it

`build_features_from_videos.py.orig` is recovered from `344e299^`, the commit
that deleted `ml/`. It is the only module there that used the live pipeline
rather than torch, and its `vectorize_shot()` is a genuinely useful starting
point.

**It does not run against today's pipeline, and it fails silently rather than
loudly.** Two things changed underneath it:

1. `KINEMATIC_FIELDS` still lists `ankle_velocity_y` and `hip_velocity_y`. Both
   were **deleted** from `KinematicFeatures` — `hip_velocity_y` was provably
   always `0.0`, because MediaPipe world landmarks are hip-centred so the hip
   midpoint *is* the origin (see the comment at `phase_detection/features.py`).
   The file reads them with `getattr(feats, field, None)`, so they do not
   raise: they become **two columns of zeros** carried into training unnoticed.

2. `PHASE_ORDER` was 8 phases and is now 6, so `FEATURE_DIM` has changed. The
   old `train.npz` and checkpoints were dimensionally incompatible and were
   deleted rather than migrated.

Copy from it deliberately, field by field. Do not import it.

## Rules this directory inherits

The same five that govern the rest of the project, and for the same reasons:

1. **Seconds, never frame counts.** Footage runs 12–60 fps and includes slow
   motion. A window in frames is a different duration on every clip.
2. **"Not observed" is never a number.** A missing landmark is `None`. Feeding
   a fabricated `0.0` into a feature vector is the same error that produced a
   whole coaching sentence from a landmark nobody saw.
3. **No per-video constants.** No `if video_name ==`, no threshold tuned until
   one clip passes.
4. **Measure before you change anything.** Including before you change a
   hyperparameter because validation looked bad once.
5. **Splits are grouped, never random.** See below — this is the one that will
   silently destroy the accuracy number.

## Why the split is grouped and not random

Three sources of leakage exist in this dataset, and a random split hits all
three:

- **Duplicates.** `assets/videos/single_shot/` was cut from `video8` and
  `video9`, which are also present whole. Clips may repeat across directories.
  The same shot on both sides of the split means the model is tested on
  something it memorised.
- **Same player.** Several clips per player (Curry, Klay, Booker, Durant,
  Salah). Split them across train and test and the model can learn the player
  — jersey, court, build — instead of the movement, and report a high accuracy
  that means nothing.
- **Same source video.** All `video8_shot*` share one gym, one camera, one
  outfit, one lighting.

So the split is by **group**, where a group is (source video, player), and
duplicate sets are pinned to a single side. A held-out test set is chosen once
and not looked at until the end.

Grouping is **transitive**, computed by union-find. An earlier version applied
the three rules in priority order and split one player across three groups; the
leakage check passed by luck, not by construction. `leakage_report()` now
refuses to write a manifest at all if any group spans a split.

A correct split can still be a useless one. The first test set held out two
whole players — sound against leakage, and silent about the fact that both were
broadcast footage with a panning camera, so five of its eight shots had no
measurable target. Both properties have to hold at once: **disjoint by group,
and measurable.**

## Why preprocessing crops but never tracks

`preprocess.py` renders a static crop around the shooter, upscaled. The crop is
computed **once per clip and never moves**, and that is the whole design.

`body_rise_ratio` and `takeoff_elevation` measure the ankle's displacement from
a standing baseline in image space. A crop that tracked the shooter frame by
frame would subtract exactly the vertical translation those features exist to
measure — and it would do it silently, turning every jump shot into a plausible
set shot rather than an obvious failure.

The crop also **preserves the source aspect ratio**, and that is load-bearing.
Landmarks are normalised per axis — x by frame width, y by frame height — and
the project compares across the two: hip travel is a fraction of WIDTH while
the body height normalising everything else is a fraction of HEIGHT. An
anisotropic crop rescales the axes differently and silently changes what those
numbers mean. Cropping `video8_shot04` to 224×343 from 1280×720 turned 0.084
frame-widths of hip travel into roughly 0.48, past the 0.18 driving threshold;
the clip was reclassified `LAYUP`, rejected, and seven siblings went with it.
The footage and the classifier were both fine.

The A/B against raw footage is in `compare_preprocessing.py`, and the number to
watch is **class changed**: of the shots detected both ways, none moved across
the jump/set boundary. That is the evidence the crop is not distorting the
elevation. If a future change to this stage makes that number non-zero, the
change is wrong no matter what it does to the totals.

Note the scope of that evidence, which is narrower than it first reads: it
covers the **elevation class only**. It says nothing about any other measure
the crop feeds — see below for one it does not cover.

### Known, not fixed: the driving test is not scale-invariant

`DRIVING_HORIZONTAL_TRAVEL_RATIO` (0.18, `shots/classifier.py`) compares hip
travel to **frame width**, while every other measurement in the project is
normalised by the player's own on-screen height precisely so it survives zoom
and camera distance. So the same physical sway reads differently depending on
how tight the camera is.

**Preprocessing therefore changes it, and aspect preservation does not help.**
Keeping the crop's aspect ratio fixes the x-versus-y comparison — hip travel
against body height — but this ratio's denominator is the frame itself, so
shrinking the frame scales it directly. Measured on `video8_shot03_set.mp4`:
0.093 raw, **0.20 cropped**, past the 0.18 gate. The clip is an
owner-labelled set shot from the fixed-camera fixture, and it is refused.

**The raw footage arbitrates.** When a *prepared* clip has a shot refused as an
unsupported type, `extract_shots.py` re-analyses the original and compares:

- raw refuses it too → the refusal is real, and it stands
- raw does not → the refusal was ours, and the raw analysis is used for that
  **whole clip**, tagged `source: "raw_fallback"` and named in the report

Whole clip, never spliced. Two analyses segment independently, so shot 1 does
not mean the same attempt in both, and mixing rows across runs would pair a
target with the wrong footage.

Mixing raw and prepared targets in one dataset is safe here, and that is
measured rather than assumed: across the 19 shots with an elevation both ways
the median difference is **0.0008** and the largest is **0.0128**, against a
jump/set boundary of 0.12 and a corpus spanning 0.003–0.274.
`takeoff_elevation` is normalised by the player's own height, which is exactly
why — it is the frame-relative measurements that cropping moves.

This does not touch the 0.18 threshold, which stays as it is.

**Measured outcome: refusals 1 → 0, usable targets 26 → 26.** The recovered
shot classifies as `set_shot`, matching its owner label, and scores 35 — so a
real misclassification is gone. It still carries no target, because its stance
was never on camera: it is a single-shot cut that opens mid-attempt, and
`takeoff_elevation` refuses to invent a floor it never saw. That was true of
the raw footage all along, so this fix was never going to add a target. It
removes a wrong answer, not a missing one.

### Scope: jump shots and set shots, nothing else

A driving action is reported `UNKNOWN` and never scored — never as a layup,
because the horizontal test measures that an attempt is *not* stationary
shooting and says nothing about what it is instead.

The gate is a **guard, not a feature**. It is tempting to read "we only support
two types" as "delete the third", but removing it would not narrow the product;
it would send drives into the jump-shot rules and produce confident, wrong
coaching. Every shot in `shots.json` is `set_shot` or `jump_shot` because of
that gate, not in spite of it.

Panning clips are **not** stabilised. Residual stabilisation error is a slow
drift, and every ratio feature here is normalised against a baseline gathered
over the clip, so a drift would bias that baseline instead of breaking it — an
error that survives review because the output still looks like data. Those
clips are marked `drop_elevation` and their target is reported missing.

## Result: no model beats predicting the mean

Four families were fitted — ridge, k-NN, boosted stumps, and a small 1-D CNN
over the sequences — against a constant predictor, with leave-one-group-out
cross-validation and the knob chosen inside each fold.

**None of them beat the constant baseline.**

```
                out-of-fold (18 dev)        held-out test (8)
constant        MAE 0.0339   acc 89%        MAE 0.0665   acc 62%
ridge           MAE 0.0372   acc 89%        MAE 0.0650   acc 62%
knn             MAE 0.0355   acc 89%        MAE 0.0638   acc 62%
stumps          MAE 0.0424   acc 89%        MAE 0.0662   acc 62%
tempnet         MAE 0.0414   acc 78%        MAE 0.0578   acc 62%
```

Every family is WORSE than the constant out-of-fold. The 62% test accuracy is
not an achievement — it is what predicting "set shot" for everything scores,
because elevation is compressed below the 0.12 boundary. The 95% interval on
test accuracy, resampling groups, is **[40%, 100%]**: eight shots from two
groups cannot resolve anything.

The models memorise instantly, exactly as the sample size predicts:

```
                    in-sample MAE      out-of-fold MAE
knn (k=3)              0.0000              0.0355
ridge (alpha=1)        0.0024              0.0372
constant               0.0306              0.0339
```

k-NN reproduces all eighteen development targets perfectly and still
generalises worse than the mean. The nested CV responded correctly by choosing
the most regularised setting available in every fold.

### The controls, including the one that failed

**Positive control (passed).** On a synthetic target built to be linear in
three of the real features, with the same 26 samples and the same group folds,
ridge halves the constant's error — 0.0222 against 0.0435. The harness learns
when there is something to learn.

**Leakage control (inconclusive, and I expected otherwise).** The first
attempt was to hand the models the excluded channels and watch the score go
near-perfect. It did not: the contaminated run scored the same as the clean
one. That is not evidence the harness is broken — `takeoff_elevation` is a
median stance baseline subtracted from a minimum inside a window around the
wrist peak, and ridge over per-channel min/max/mean cannot express that
whichever channels it gets. The control tested the wrong thing, which is why
the synthetic target exists.

The exclusion list stays regardless. It is justified by what the target IS,
not by a score.

### So: data shortage, and it is measurable

18 development samples and 8 test samples. The corpus has 43 shots but only 26
carry a target — **17 are lost to measurement, not to labelling**:

```
10   camera pans: player rise is not separable
 7   no stance before the shot, or the feet were never seen
```

Recovering those is worth more than any modelling change. Ten of them need
stabilisation or a different elevation estimator; seven need footage that
starts before the shot rather than at it. That alone would take 26 targets to
43, and the second-largest group in the split from 2 shots to something that
can validate.

## How accuracy is measured

`evaluate.py` takes `{clip#shot: predicted_elevation}` and the frozen split. It
was written **before any model existed**, so no choice in it was made after
seeing a result.

It reports four things a bare accuracy number hides:

1. **The training-mean baseline.** Elevation on this corpus is compressed
   around the 0.12 boundary, so a constant predictor already scores well —
   measured at **92.9% on train**. Any model that does not clearly beat that
   has learned the mean, not the movement.
2. **The ambiguous band held apart.** Labels contradict each other between 0.05
   and 0.15. Averaging those in mixes "did it learn shooting" with "did it
   guess our threshold", and only the first is a skill.
3. **The train–test gap**, because twenty-odd samples will be memorised by
   anything with capacity.
4. **A bootstrap interval resampled over GROUPS, not shots.** Five of Salah's
   shots are not five independent observations. The resulting interval is wide;
   that width is the finding, not a defect in the estimate.
