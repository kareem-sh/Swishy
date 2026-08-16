# Swichy — basketball shooting coach

Upload a video of someone shooting. Get their phases, a score, and what to fix.

Single camera, MediaPipe pose, no markers, no sensors.

```bash
python main.py                                    # the video set in main.py
python main.py assets/videos/video_01_free_throw.mp4  # or any file
```

It prints a report, then replays the video with the analysis drawn on it.

---

## Offline, not live

The whole file is read first. Only then is it analysed. That is not a
performance compromise — it is what makes the analysis correct.

A shot is a shape in time: the hand rises out of a stance and comes back down.
Deciding at frame 400 whether frame 400 belongs to a shot means guessing what
frames 401–460 will do. Everything that used to guess — timeouts, hysteresis,
latches, refractory timers — existed only to cover that guess, and none of it
survives once the file has been read.

The same applies to the drawing. A frame's coaching phase depends on where the
knee bottomed out and where the hand peaked, neither of which is known while
that frame is on screen. A live overlay can only show a guess; this one shows
the answer.

Live webcam capture still exists in `modes/live_stream.py`. It cannot use this
path and is not the product target.

---

## What the footage needs

**Start recording with the player standing still, at least ~0.8 s before the
shot begins.**

That number is measured, not chosen. Cutting a clip at the release costs both
measurements at once:

| Needs the frames before the shot | Because |
|---|---|
| Finding the shot | Prominence measures the peak against the valley beside it |
| Jump vs set shot | Foot rise is measured against where the feet were standing |

On five real attempts, detection went 2/5 at 0.5 s of lead-in and 5/5 at 0.8 s.
More than that adds nothing.

Other requirements: whole body in frame, one player, camera roughly still.
Broadcast footage that zooms and pans is out of scope — measured, it zoomed
4.7× in 2.5 s, which reads as the player rising a metre.

---

## How it works

```
video
  │  pass 1 — read and measure, decide nothing
  ├─ pose/          MediaPipe landmarks, Y flipped once to +Y up
  ├─ filters/       One Euro smoothing
  ├─ angles/        3D joint angles
  └─ phase_detection/features.py
        wrist_height_ratio, body_rise_ratio  (image space: world
        landmarks are hip-centred and blind to whole-body motion)
  │
  │  pass 2 — decide, with the whole signal in hand
  ├─ shots/segmenter.py    find shots by prominence, bound them where
  │                        the hand was last down
  ├─ shots/elevation.py    jump vs set, against the player's own stance
  ├─ feedback/phase_refiner.py   7 coaching phases, by extrema
  ├─ analysis/engine.py    13 rules from config/biomechanics.yaml
  └─ feedback/scorer.py    per-phase and overall score
  │
  └─ modes/replay.py       redraw the video with the final labels
```

`ball/` and `mlRingBall/` provide ball and rim detection. The ball state
machine confirms release from measured possession and flight, then determines
the made/missed outcome used by the offline report.

---

## Documentation

| File | What is in it |
|---|---|
| [docs/SEGMENTATION.md](docs/SEGMENTATION.md) | How shots are found and bounded, with the measurements |
| [docs/PHASES_AND_SCORING.md](docs/PHASES_AND_SCORING.md) | The 7 phases, the 13 rules, how a score is built |
| [docs/LIMITS.md](docs/LIMITS.md) | What this does not do, and what is known to fail |
| [SOURCES.md](SOURCES.md) | Every research claim, its evidence class, and a link |

---

## Tests

```bash
.\.venv\Scripts\python.exe -m pytest tests -q
```

The venv matters — system Python has no `cv2`.

`tests/test_acceptance_videos.py` runs real videos through the real pipeline.
Known failures are marked `xfail(strict=True)` with the measured reason, so a
limitation cannot quietly become a regression.

---

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The pose model is expected at `models/pose_landmarker_full.task`.
