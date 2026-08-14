# Phases and scoring

## Two layers, and why they are separate

```
DETECTOR    4 states, live        phase_detection/detector.py
            ready → rise → release → recovery

ANALYSIS    7 phases, after       feedback/phase_refiner.py
            ready_stance · loading · ball_lift · jump ·
            release · follow_through · landing
```

Finding a shot wants **few** states — every state is another place to get stuck
and lose the shot. Coaching wants **many** phases — "your knee bend" and "your
follow-through" are different advice under different headings.

They used to be one thing: an eight-state machine whose state was also the
coaching label. Reaching `release` meant passing seven consecutive threshold
gates, and any one failing lost the whole shot. Measured: 4 of 10 on a
ten-shot video, 0 of 1 on a side-on jump shot.

The analysis phases are now found **by extrema over the captured window** — the
frame where the knee angle was smallest is the bottom of the dip. An `argmin`
over an array cannot get stuck. A live velocity threshold can, and did.

> The refiner takes the shooting event from the segmenter, not from the
> detector's labels. It used to read `phase == "release"` off the frames — the
> old detector's output — inside the path built to replace it. Where that
> detector had entered `release` early, the dip and the lift had nowhere to
> live and a 2.4 s shot reported three phases. Passing the real peak moved
> scores from 12–60 to 41–68.

---

## The seven phases

| # | Phase | Located by | Rules |
|---|---|---|---:|
| 1 | `ready_stance` | before the knee starts to bend | 1 |
| 2 | `loading` | `argmin` of knee angle = bottom of the dip | 4 |
| 3 | `ball_lift` | dip bottom → take-off | 4 |
| 4 | `jump` | body rise ≥ 0.05 body heights (absent for a set shot) | 3 |
| 5 | `release` | the shooting event, from the segmenter | **6** |
| 6 | `follow_through` | release → hand drops | 2 |
| 7 | `landing` | ankles back to baseline | 1 |

`release` carries almost half the rules, which matches the evidence: release
height is the strongest proficiency correlate in `SOURCES.md`.

`knee_flexion` was removed. It carried the same two rules as `loading`, so one
dip was reported twice under two headings, sometimes with contradictory
readings because the deepest bend fell in one and a shallower one in the other.

### Phases without their own rules

`ready_stance` and `jump` have **no rule unique to them** — everything they
measure (`trunk_posture`, `head_stability`, `shoulder_alignment_lift`) is also
measured elsewhere. That is the same argument that justified deleting
`knee_flexion`, and merging them would shorten the report without losing a
measurement.

Not done yet, deliberately: after segmentation was fixed, phase capture went
from 22/35 to 31/35 on `salah_video`, so this is now a clarity change rather
than a fix, and `landing` — which *would* be merged away — carries the only
balance rule.

---

## Scoring

`feedback/scorer.py`, weights in `config/scoring.yaml`.

Rules are weighted by severity: `error` 3, `warning` 2, `info` 1. Each phase is
scored from the rules that actually fired in it, and the overall score is their
weighted combination.

**A phase that was not captured contributes nothing.** This is why segmentation
governs everything downstream: a shot bounded at 0.63 s captures `release`
alone, scores 6 of 13 rules, and reports a number that looks like a judgement
of the player but is a judgement of the window.

Two rules are **measured and displayed but not scored** — the shoulder
elevation rules. MediaPipe's shoulder-angle error reaches 26° depending on
camera azimuth (`SOURCES.md` §D4), which is larger than the difference being
measured. Showing them is honest; scoring them would not be.

---

## Shot type

`shots/elevation.py` and `shots/classifier.py`.

How far the feet rose at the shooting event, in body heights, against a
threshold of **0.12** — from the literature's 15.3 ± 5.1 cm for free throws
versus 26.9–31.2 cm for jump shots.

The threshold was right all along. What was wrong was the reference under it.

**The baseline is local, taken from the player's own stance seconds before the
shot.** The floor is not a horizontal line in an image — it is a plane in
perspective, so a player standing further from the camera has their feet higher
in frame while standing on the same floor. A whole-clip reference is only valid
while the player never moves, and in practice footage they walk out to collect
the ball between every rep. With a global baseline, real jump shots measured
*negative* elevation.

| | global baseline | **local baseline** |
|---|---|---|
| `video8` set shots | 0.025 – 0.051 | 0.015 – 0.100 |
| `video8` jump shot | 0.075 ✗ | **0.181** ✓ |

14 of 14 correct on fixed-camera footage. The margin above the highest set shot
is **0.019** — thin — and only one labelled jump shot has been measured this
way. Set-shot detection is well evidenced; jump detection is not yet.

The toe was tried instead of the ankle, on the sound reasoning that rising onto
the balls of the feet lifts the ankle without the foot leaving the floor. It
measured **worse**: 0.001 of headroom against the ankle's 0.019, because
`foot_index` is among MediaPipe's least stable landmarks. The theoretical gain
was swallowed by measurement noise.
