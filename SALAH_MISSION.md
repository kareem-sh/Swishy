# SALAH MISSION — ball integration and ball-side rules

You own `ball/`. I own `phase_detection/`, `feedback/`, `analysis/`, `angles/`.
This file is the contract between them: what the phases are now, how each
boundary is computed, where the ball plugs in, and how to add rules that score
the ball.

Read §0 before anything else. It is short and it is the part that will cost you
weeks if you skip it.

---

## 0. FIVE RULES THAT ARE NOT NEGOTIABLE

These are not style preferences. Every one of them was written after a bug that
took days to find.

**0.1 — SECONDS, NEVER FRAME COUNTS.** Our footage runs 12–30 fps and includes
slow motion. `if frames_since_release > 5` is four different durations
depending on the clip. Every window, timeout and threshold is in seconds,
derived from `timestamp_ms`. There is one known violation left, in your code:
`ball/shot_state_machine.py:404` uses a per-frame pixel delta. It will behave
differently on 12 fps and 30 fps footage.

**0.2 — "NOT OBSERVED" IS NEVER A NUMBER.** If the ball was not detected, the
answer is `None`, not `0.0`, not the last known position, not an interpolation.
This project has been bitten by this three separate times, most expensively
when an ungated wrist landmark read `0.0` in hip-centred coordinates — which
means "exactly at hip height", a perfectly plausible value that nothing
downstream could question. A whole coaching sentence was generated from a
landmark nobody saw.

If your detector loses the ball, say so. A rule that receives `None` is
skipped, and the report tells the player the recording was the problem. That is
a correct outcome. A fabricated value is not.

**0.3 — NO PER-VIDEO ANYTHING.** No `if video_name ==`, no `if fps ==`, no
threshold tuned until one fixture passes. If a constant only works on one clip
it is not a constant, it is a bug with good manners.

**0.4 — MEASURE BEFORE YOU CHANGE A BAND.** Today I was certain that a rule had
started measuring something new, wrote the reasoning out, and the measurement
refuted it in one line: two rules I thought were different had a median
difference of 2.3° and were the same number twice. If I had "fixed" the failure
rate by widening a band instead of measuring, the duplication would have been
hidden behind a better-looking score forever.

Widening a band until your data passes is the single easiest way to destroy
this project quietly.

**0.5 — LABEL YOUR EVIDENCE.** Every rule carries an `evidence:` block naming
its class: `DIRECTLY SUPPORTED` / `SUPPORTED WITH LIMITATIONS` / `INFERRED` /
`ENGINEERING DECISION` / `NOT SUPPORTED`. If you chose a number yourself, say
`ENGINEERING DECISION` and say so plainly. Do not dress your own choice as
science — several rules in `config/biomechanics.yaml` carry warnings because
someone once did, including a citation with a p-value that appears in no paper
we hold.

---

## 1. WHAT CHANGED IN THE PHASES

### 1.1 There are two phase vocabularies, not one

| layer | where | what it is |
|---|---|---|
| `CORE_STATES` | `phase_detection/detector.py` | 4 live states: `ready`, `rise`, `release`, `recovery` |
| `PHASE_ORDER` | `feedback/phase_refiner.py` | 6 coaching phases, assigned **after** the shot is captured |

Both are defined in `config/phase_model.yaml`. Nothing is hardcoded.

**The coaching phases are not produced by the state machine.** They are found
afterwards, by looking at the whole shot at once. That is the important change:
live detection has to decide "is the knee bending NOW?" from a velocity and a
threshold, and if it gets that wrong it is stuck in the wrong state for the
rest of the shot. Looking backwards, the same question is "which frame had the
smallest knee angle?" — an `argmin` over an array. It cannot get stuck and it
needs no threshold.

The six coaching phases:

```
loading  ->  ball_lift  ->  jump  ->  release  ->  follow_through  ->  landing
                             ^ omitted entirely when the feet never left the floor
```

### 1.2 The equation of each phase

All of this is `feedback/phase_refiner.compute_cuts()`. `n` is the frame count
of the captured shot. `event_index` is the shooting event, located by the
offline segmenter as the peak of the wrist's own trajectory across the whole
video.

**Signals it uses** (all per frame):

```
wrist[i]  = features.wrist_height_ratio      # image space, preferred
            or features.wrist_y              # world, only if the gate passed it
knee[i]   = features.knee_angle              # after _mask_impossible(), see below
rise[i]   = features.body_rise_ratio         # image space, body heights
```

**The cut points, in order of derivation:**

```
release_start, release_end = span of ±0.15 s around event_index,
                             walked by TIMESTAMP not by frame count

dip_bottom   = argmin(knee, 0, release_start)          # deepest bend before the ball goes
lift_start   = dip_bottom + 1                          # +1 so the deepest bend stays IN loading
load_start   = argmax(knee, 0, dip_bottom + 1)         # straightest knee on the way in

peak_rise    = max(rise[i]) for i in [lift_start, n)
jumped       = peak_rise >= 0.05                       # min_takeoff_ratio, phase_model.yaml
takeoff      = first i in [lift_start, release_start) where rise[i] >= 0.35 * peak_rise
               (= release_start when jumped is False, which collapses `jump` to nothing)

landing_start = first i after release_end where, and at least 0.15 s after it:
                  jumped      -> rise[i] < 0.35 * peak_rise      (feet back down)
                  not jumped  -> the shooting hand is back below the shoulder
```

Then **all six cut points are forced non-decreasing**. A signal that reads out
of order collapses its own segment to zero width rather than scrambling every
segment after it.

**The labels that fall out:**

| phase | frames | what defines it |
|---|---|---|
| `loading` | `[0, lift_start)` | everything up to and including the dip bottom |
| `ball_lift` | `[lift_start, takeoff)` | upward drive, feet still down |
| `jump` | `[takeoff, release_start)` | airborne — **empty for a set shot** |
| `release` | `[release_start, release_end)` | ±0.15 s around the located event |
| `follow_through` | `[release_end, landing_start)` | the finish |
| `landing` | `[landing_start, n)` | back down |

### 1.3 Two changes that will surprise you

**`release` used to be exactly ONE frame.** `_release_span` returned
`(event, event+1)`, so `aggregate: max` on the release rules was not an
aggregation — it was a single sample, with no protection against the event
being located a frame or two early. Measured across 46 shots,
`elbow_extension_release` cleared its floor on 26 of them and reported values
like 79°, which is not a release, it is an arm still folded mid-lift.

It is now ±0.15 s. That width is **our uncertainty about when the release
happened**, not a claim about how long a ball takes to leave a hand. Widths
measured:

```
±0.00s  57%     ±0.05s  67%     ±0.10s  76%     ±0.15s  80%     ±0.20s  80%
```

The plateau is the evidence, not the peak. A window drifting into the arm's
descent would keep finding larger angles as it widened; this one stops.

**This matters directly to you**: when you sync ball release to pose release,
`release` is now a window of several frames, not a point. Use `event_index`
if you need the point.

**Knee readings are masked before the cuts are computed.** `loading` ends at
the smallest knee angle in the shot — so a frame whose knee is wrong by 100°
does not merely get scored, it *becomes* the phase boundary. On one clip the
knee read 139.2 → 31.9 → 80.4 on consecutive frames and `loading` was cut at
the 31.9.

`_mask_impossible()` removes frames that stray off the line between their
neighbours by more than the joint could have travelled *and returned* in the
shorter of the two gaps. Note that a plain rate test does **not** catch this:
the corrupt frame arrived after a 133 ms tracking gap, across which 107° is
genuinely possible, so degrees-per-second read 806 and called it plausible. It
is the *shape* that gives it away, not the speed. You will need the same idea
for ball position after a detection dropout.

### 1.4 A phase that happened always gets a mark

Three distinct states, three different words in the report:

| shown | meaning |
|---|---|
| phase absent from the report entirely | it did not happen |
| `-- not applicable` | it happened, nothing applies by design (`jump` in a set shot) |
| `-- not filmed clearly` | **it happened and we could not measure it** |

The third prints the reason and names the missing measurement:

```
Ball Lift    --    not filmed clearly
  [FILMING]  Ball Lift happened, but Ball Set Position could not be measured
             anywhere in it. That is a limit of this recording, not of the shot.
```

If your ball rules can't measure something, produce `None` and this handles it.
Do not invent a value to avoid a blank.

---

## 2. HOW A RULE WORKS

Everything lives in `config/biomechanics.yaml`. `analysis/engine.py` reads it.
Adding a rule normally means editing YAML and adding one metric.

### 2.1 Every field

```yaml
  my_rule_id:
    name: "What The Player Sees"
    metric: my_metric            # the key in BiomechanicsEngine._measure
    phases: [release]            # which coaching phases it runs in
    shot_types: [jump_shot]      # OPTIONAL. omit = every shot type
    aggregate: max               # min | max | worst   -- see 2.2
    max_rate: 2000               # OPTIONAL. units-per-SECOND physical ceiling
    min: 142                     # outside min..max        -> NEEDS WORK
    max: 180
    ideal_min: 152               # inside ideal band       -> EXCELLENT
    ideal_max: 168               # inside min..max only    -> GOOD
    unit: "°"
    severity: warning            # error | warning | info  -> scoring weight
    scored: true                 # false = measured and displayed, never scored
    message_excellent: "..."
    refine_low: "..."            # GOOD tier, value BELOW the ideal band
    refine_high: "..."           # GOOD tier, value ABOVE it
    message_low: "..."           # NEEDS WORK, below min
    message_high: "..."          # NEEDS WORK, above max
    evidence: >
      ENGINEERING DECISION / DIRECTLY SUPPORTED / ...
```

Messages use **external focus** — point at the ball, the rim, the floor, not at
the limb. "Send the ball up off the floor" beats "bend your knees". This is one
of the most replicated findings in motor learning and it is tested on
basketball free throws specifically.

The GOOD tier is directional on purpose: a value under the ideal band needs the
opposite correction from one over it, so a single "refine this" message would
be useless or actively wrong.

### 2.2 `aggregate` — which frame of the phase represents the rule

This is the field people get wrong.

| policy | question it answers | use for |
|---|---|---|
| `min` | how deep did it go | the dip, the set point |
| `max` | how far did it reach | extension, the held finish |
| `worst` | did it ever slip | continuous rules: posture, alignment |

`worst` is the default, so a rule that declares nothing keeps the old
behaviour rather than silently changing meaning.

Getting this wrong is not subtle. Before it existed, the knee rule took the
worst frame of `loading` — but the phase *begins* at the straightest knee by
construction, so the worst frame was always the first one. It reported 165°
when the player's actual load was 91° and told them to bend deeper. The advice
was not merely wrong, it was reversed.

For the ball, expect: `max` for apex height and entry angle, `min` for distance
from the shooting hand at the set point, `worst` for anything continuous like
lateral deviation.

### 2.3 `max_rate` — the despike guard

Optional, in **units per second**. A rule that declares it gets lone corrupt
frames dropped before aggregation (`feedback/scorer._despike`). A rule that
declares nothing is untouched.

This matters most for `min`/`max` rules, because they seek the extreme value —
so a single corrupt frame is exactly the frame they select.

**For ball rules this is close to mandatory.** A YOLO false positive on a head,
a shoe or a light fitting produces exactly the jump-away-and-back shape the
guard is built for.

### 2.4 Shot-level rules — the pattern you will need

Most rules are per frame. Some are not: "how long was the finish held" cannot
be a per-frame metric, because the engine sees one frame at a time and has no
idea when the release happened.

`follow_through_hold` is the worked example. Read it:

- `feedback/phase_refiner.hold_duration_s()` measures it once per shot
- `feedback/shot_tracker._finalize()` passes it into `score_shot(hold_s=...)`
- `feedback/scorer._hold_rule()` turns it into a `RuleResult` using the bands,
  messages and severity from the YAML like any other rule
- it returns `None` when the hand never came down inside the clip

**Every ball outcome rule is this shape**: entry angle, make/miss, apex height,
release-to-rim time. They are all one-per-shot. Copy `_hold_rule` and pass your
value the same way.

---

## 3. WHERE THE BALL PLUGS IN

### 3.1 What already exists

| file | what it gives you |
|---|---|
| `ball/models.py` | `BallDetection`, `RimDetection`, `BallSnapshot`, `BallTrajectory`, `ShotOutcome` |
| `ball/timeseries.py` | `BallTimeSeriesBuffer` — the per-frame history |
| `ball/shot_state_machine.py` | `in_hand` / `in_flight` / `at_rim` |
| `ball/release_sync.py` | `ReleaseSync.find_release_frame()` — aligns ball release with pose release |
| `pipeline.py` | holds `_ball_buffer`, `_ball_shot_fsm`, `_ball_tracker`, and passes `enable_ball` |

`ShotSummary.outcome` is already a field. It is wired and unused for scoring.

### 3.2 The one thing that matters most: WHOSE release wins

You have two independent estimates of when the ball left the hand:

```
pose  : event_index  = peak of the wrist trajectory  (shots/segmenter.py)
ball  : ReleaseSync.find_release_frame()             (ball/release_sync.py)
```

**Do not average them and do not silently prefer one.** Report both and their
disagreement. A large disagreement is a *measurement*, and a useful one — it
means either the ball detector or the pose is wrong on this clip, and the
report should say the shot was measured with low confidence rather than pick a
winner.

If you must pick one for the phase cuts: **keep pose as the authority**, because
every existing phase boundary and every existing band was derived against it.
Changing the anchor changes every score in the project. If ball release turns
out to be better, that is a real finding — measure it against the labelled
clips first, show the difference, and change it deliberately.

### 3.3 Suggested integration order

**Step 1 — make ball data reach the engine.** Today `_measure` receives
`(metric, angles, features, shooting_side)`. Add ball the same way `shot_type`
was added this week:

```python
# analysis/engine.py
def evaluate(self, phase, angles, features, shooting_side,
             shot_type=None, ball=None):
    ...

def _measure(self, metric, angles, features, shooting_side, ball=None):
    if metric == "ball_wrist_distance":
        if ball is None or ball.center_xy is None:
            return None                      # NOT 0.0 -- see rule 0.2
        return distance_in_body_heights(ball, features)
```

Then thread it from `feedback/shot_tracker._refined_frames()`, which is where
`shot_type` is threaded now. Follow that diff exactly.

**Step 2 — normalise ball distances the way we normalise everything else.**
Pixels are not comparable across clips: camera distance, zoom and resolution
all change them. Divide by the player's on-screen height, the same denominator
`wrist_height_ratio` and `body_rise_ratio` already use. That is what makes a
number comparable between a phone video and a broadcast clip, and it needs no
player height entered.

**Step 3 — shot-level outcome rules**, via the `_hold_rule` pattern in §2.4.

**Step 4 — only then**, consider using ball state to refine phase boundaries.
Not before. The pose phases work today and are validated; do not put them
downstream of a detector that is still being built.

### 3.4 Ball rules worth adding, in the order I would do them

| rule | phase | metric | why first |
|---|---|---|---|
| `ball_set_position` | `ball_lift` | ball-to-shoulder distance ÷ body height | the set point is currently inferred from the elbow alone |
| `release_ball_height` | `release` | ball centre vs nose, ÷ body height | the literature measures the BALL; we measure the WRIST, about 0.10 body heights lower |
| `entry_angle` | shot-level | `BallTrajectory.entry_angle_deg` | the corpus has real numbers here — A6 reports 54.5–57.9° launch |
| `ball_path_deviation` | `release`→apex | lateral drift ÷ body height | left/right miss is the one error pose cannot see at all |

**`entry_angle` is your strongest card.** It is the one ball measurement with
published values in `SOURCES.md` that we currently cannot produce at all, and
it is genuinely independent of everything pose measures.

Note on `release_ball_height`: `release_height_ratio` is currently
`scored: false` because it failed 15 of 15 attempts — a hip-centred,
non-same-frame subtraction. Its evidence block explains what a correct rebuild
needs. **If you can give a real ball position at release, you can rebuild that
rule properly and it becomes yours.** That is the single biggest thing the ball
unlocks.

---

## 4. CURRENT STATE — what you are building on

Measured across 53 clips: `assets/videos/single_shot`, `salah_video.mp4`,
`assets/videos/ShootingVideosDataset`.

```
46 scored shots      mean 77.1      min 31      max 100
 7 rejected clips    all no_shooting_event (they start mid-shot)
```

Rule failure rates:

```
jump_elevation           38%   (6/16)    band on 16 points, weakest rule here
elbow_slot_ball_lift     21%   (6/29)
elbow_extension_release  20%   (9/45)
follow_through_hold      19%   (8/43)
landing_balance          14%   (5/37)
hip_hinge_loading        13%   (6/46)
trunk_posture             2%   (1/41)
release_height            2%   (1/45)
```

### 4.1 Known limits — please do not rediscover these as bugs

**The shot classifier is 74%, not 100%.** 17/23 against filename labels, and
**all six errors are in one direction**: a jump shot called a set shot.
`shots/elevation.py` records 14/14 on our own static-camera footage; the gap is
broadcast footage with camera motion. Your ball trajectory may be able to fix
this — a ball that leaves the hand well above the player's standing reach is
evidence of a jump that `body_rise_ratio` cannot see when the camera pans.

**`body_rise_ratio` saturates on broadcast clips.** It measures the player
against the frame, so it cannot separate the player rising from the camera
falling. 4 of 53 clips hit the cap, including a *free throw* that read a full
0.500 body heights. Readings above 0.40 are now refused as unmeasured rather
than scored.

**Sample size is unhandled and it is the largest remaining gap.** A phase
measured from ONE frame is scored with the same confidence as one measured from
twenty-five. Of the nine hip failures at the old bounds, four came from phases
of five frames or fewer and three from a single frame — including the highest
reading in the whole corpus, 174.5°, from one frame. Every well-sampled shot
(12–32 frames) fell between 90.9 and 150.8 with no outlier at all.

This is the next thing being fixed on my side. Your ball rules will have
exactly the same problem, so do not build around its absence.

**Nine tests fail and five of them are yours** — the `ball/` rim-crossing
tests in `tests/test_ball_shot_state_machine.py`. They were failing before this
week's work and I have not touched them. The other four are pose-side and
tracked.

---

## 5. QUICK REFERENCE

```
config/biomechanics.yaml     every rule, every band, every message
config/phase_model.yaml      the phase vocabulary and the refinement constants
analysis/engine.py           _measure() -- add your metric here
feedback/phase_refiner.py    compute_cuts() -- the phase equations
feedback/scorer.py           aggregation, despike, _hold_rule pattern
feedback/shot_tracker.py     _refined_frames() -- where shot_type is threaded
SOURCES.md                   the evidence corpus. Cite it or say ENGINEERING DECISION.
docs/LIMITS.md               measured uncertainty of every angle we produce
```

Run one clip:

```
python main.py path/to/clip.mp4
```

Before and after any change, run the whole corpus and compare. Not one clip.
Every mistake described in this file passed on the clip it was developed
against.
