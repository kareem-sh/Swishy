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
| 1 | `ready_stance` | before the knee starts to bend | 0 |
| 2 | `loading` | `argmin` of knee angle = bottom of the dip | 3 |
| 3 | `ball_lift` | dip bottom → take-off | 3 |
| 4 | `jump` | body rise ≥ 0.05 body heights (absent for a set shot) | 2 |
| 5 | `release` | the shooting event, from the segmenter | **6** |
| 6 | `follow_through` | release → hand drops | 2 |
| 7 | `landing` | ankles back to baseline | 1 |

Counted from `phases:` in `config/biomechanics.yaml`. The first four dropped by
one when `trunk_posture` was narrowed to `release` alone — it had been running
under a single threshold across four postures, and in `loading` it penalised the
same forward hinge that `hip_hinge_loading` rewards.

`release` carries almost half the rules. That reflects where the evidence is
thickest, not a proven ranking: release height is the only variable that
separated proficient from non-proficient shooters in the one study that tested
it (`SOURCES.md` A2, n=34, single laboratory, no independent replication).

`knee_flexion` was removed. It carried the same two rules as `loading`, so one
dip was reported twice under two headings, sometimes with contradictory
readings because the deepest bend fell in one and a shallower one in the other.

### Phases without their own rules

`ready_stance` now has **no rule at all**, and `jump` has **no rule unique to
it** — both of the rules it evaluates (`head_stability`,
`shoulder_alignment_lift`) are also evaluated elsewhere, and both are unscored.
That is the same argument that justified deleting `knee_flexion`, and merging
them would shorten the report without losing a measurement.

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

Four rules are **measured and displayed but not scored**:
`shoulder_alignment_lift`, `head_stability`, `index_alignment_release` and
`follow_through_index`. Each is withheld for a measured reason, recorded in its
`evidence:` block — MediaPipe's shoulder-angle error reaches 26° depending on
camera azimuth (`SOURCES.md` §D4), larger than the difference being measured;
the two finger-line rules carry 39° of model-to-model disagreement against an
18–20° band; `head_velocity` is measured relative to the hips rather than to the
scene. Showing them is honest; scoring them would not be.

One rule is scored **against this project's own evidence map**:
`knee_flexion_loading` is `scored: true` while `SOURCES.md` Part E and §D3 both
say it must be demoted to a displayed metric. The conflict is written into that
rule's `evidence:` block rather than resolved silently, because flipping the
flag changes every score produced so far. Until it is settled, that rule's
contribution to the score has no evidential basis.

---

## Shot type

`shots/elevation.py` and `shots/classifier.py`.

How far the feet rose at the shooting event, in body heights, against a
threshold of **0.12**.

**0.12 is calibrated, not derived.** It is an engineering decision measured
against this project's footage, and `shots/classifier.py` labels it that way at
the constant itself. The literature only sets its order of magnitude: A1 reports
vertical displacement of 15.3 ± 5.1 cm for free throws against 26.9–31.2 cm for
jump shots, which for a ~1.8 m player is roughly 0.085 versus 0.15–0.17 body
heights, so the classes separate somewhere near 0.12. That study measured
posterior-calcaneus-to-ground while we measure ankle-landmark rise above a
standing baseline — related quantities, not interchangeable ones, so no
threshold can be read off it.

Changing the baseline did not vindicate the number; it left the number alone and
fixed the reference underneath it, after which the number worked on our clips.

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
