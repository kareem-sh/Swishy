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
  reference/     code recovered from the deleted ml/ dir, kept for reading only
  data/          generated feature arrays and manifests   (gitignored)
  checkpoints/   trained weights                          (gitignored)
  artifacts/     evaluation output, confusion matrices    (gitignored)
```

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
