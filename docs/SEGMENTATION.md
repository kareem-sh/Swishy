# Finding the shots

`shots/segmenter.py`. Every number here was measured on labelled footage;
`scripts/peak_probe.py` reproduces it.

---

## The signal

`wrist_height_ratio` — the shooting wrist's height above the hips, divided by
the player's own on-screen height.

Three properties earn it the job:

- **Image space, not world space.** MediaPipe's world landmarks are
  *hip-centred*, so when a player jumps, hips and ankles rise together and
  nothing changes. A real jump measured 0.030 m of "ankle rise" in world space.
- **Divided by body height**, so it is invariant to zoom and camera distance.
- **Survives the visibility gate.** A rejected world-space wrist silently reads
  0.0, which in hip-centred coordinates means "wrist exactly at hip height" — a
  perfectly plausible number. Footage shot from behind triggered that on every
  frame of a clip, and the old detector saw a motionless wrist through an
  entire jump shot.

---

## Finding a shot: prominence

A height threshold asks *"is the hand above X?"*. That is equally true of a
shot, a catch, a stretch and a scratch.

Prominence asks *"how far did the hand rise above its own surroundings?"* — the
drop you would have to descend before you could climb to any higher peak. It
has no zero point and no baseline, so a drifting signal cannot fool it.

```
                ▲ 0.69  shot
               ╱ ╲
      ▲ 0.45  ╱   ╲          prominence of the shot = 0.69 − 0.03 = 0.66
     ╱ ╲_____╱     ╲___       prominence of the bump = 0.45 − 0.35 = 0.10
____╱  0.35        0.03
```

### Why the threshold is 0.55

Measured on `video8`, whose ten shot windows are frame-exact in
`assets/videos/single_shot/manifest.json`:

| | prominence |
|---|---|
| The nine visible shots | **0.636 – 0.705** |
| The loudest non-shot | **0.445** |

Any threshold from 0.45 to 0.63 returns the identical answer. 0.55 is the
middle of that basin, not the edge of a cliff — which is the difference between
measuring a phenomenon and memorising a file.

Two peaks closer than **1.0 s** collapse to the more prominent one. Nobody
shoots twice inside a second.

---

## Bounding a shot

Not with `peak_widths`. That descends a share of the *prominence* and runs
outward until the signal crosses it. Between reps the player walks back holding
the ball, the signal never returns that low, and windows of **25 s** were
measured for 2 s shots.

Descending a share of *this peak's own rise* gives a level the signal must have
crossed, because the peak rose from it. Two guards sit on top:

| Guard | Why |
|---|---|
| Absolute cut at wrist ≈ hip height | A shorter peak yields a higher relative cut; on footage where the player swings their arms walking, the window closed on a passing bump. Two of five attempts bounded at 0.63 s and 0.73 s. |
| Minimum window 1.5 s | A shot cycle does not happen in less. Can only widen a window, never narrow one. |

Result on `video8`: windows of 1.87–2.40 s, mean IoU **0.84** against the
frame-exact truth, one window per real shot.

---

## Short clips take a different question

A clip cut tightly around one shot has no valley on one side, so its prominence
collapses. Measured: 7 of 13 single-shot clips found nothing at 0.55.

On a clip of **8 s or less** the question is no longer *"which of these bumps is
the shot"* — there is only one candidate — it is *"is this a shot at all"*. That
gets a lower bar (0.25) applied to the single best peak, so it can rescue one
attempt but never invent several.

It is still a real bar. A clip of someone walking past must find nothing.

---

## Measured results

| Footage | Attempts | Types | Note |
|---|---|---|---|
| `salah_video` | **5 / 5** | 5 / 5 | live path found 11 candidates, 3 of them real |
| `video8` | 9 / 10 | **9 / 9** | miss is shot 1, which starts at frame 4 |
| single-shot clips | 10 / 13 | — | the 3 misses are cut at the release |
| `Couch*` (broadcast) | 4 / 7 | 1 / 3 | every clip ≤ 2.0 s failed; every clip ≥ 2.3 s was found |

---

## What it cannot do

**Tell a shot from a catch.** Both are the hand going up and coming down. The
body does not distinguish them; the ball's direction of travel does, which is
what `shots/ball_check.py` is for.

**Find a shot already in progress when recording starts.** No rise from rest
means no prominence. This is a truncation case, not a threshold case, and more
lead-in beyond ~0.8 s does not help because the valley is either in frame or it
is not.
