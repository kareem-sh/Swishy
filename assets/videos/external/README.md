# External test videos

Downloaded for **generalization testing only**. These are NOT canonical fixtures
and are NOT tuning targets. The fixture videos in `assets/videos/` are untouched.

`.gitignore` excludes video files from the repo, so these are not committed.
Re-download with the commands below.

## Scope

Swichy analyses **free throws, set shots and jump shots** (see
`docs/GENERALIZATION_BASELINE.md` §Scope). Layups, hook shots, dunks and tip-ins
have different mechanics and are **out of scope**. The layup below is kept
deliberately as a **negative control**: it tests whether the system correctly
refuses an unsupported shot type rather than silently scoring it.

## Files

| File | Action type | In scope? | Source | Licence | Duration / FPS |
|---|---|---|---|---|---|
| `external_01_free_throw_curry_kerr.mp4` | Free throw (set shot) | ✅ Yes | [YouTube `b2E7kCHkSO0`](https://www.youtube.com/watch?v=b2E7kCHkSO0) | Creative Commons Attribution (reuse allowed) | 14.75 s @ 29.97 fps, 640×360 |
| `external_02_jump_shot_peja.mp4` | Jump shot (repeated reps) | ✅ Yes | [YouTube `Bf7NvlRRKY4`](https://www.youtube.com/watch?v=Bf7NvlRRKY4) | Creative Commons Attribution (reuse allowed) | 50.25 s @ **12.00 fps**, 320×240 |
| `negative_control_layup_van_rossom.webm` | Layup | ❌ No — negative control | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Safata_Van_Rossom.webm) | CC BY-SA 3.0, by Coentor | 9.10 s, 640×360, **container reports 1000 fps** |

## Why each was selected

- **`external_01`** — an in-scope free throw from a source outside our fixtures, at a different resolution and framing. Tests whether free-throw segmentation generalises beyond `video_01`.
- **`external_02`** — an in-scope jump shot with **repeated reps** at **12 fps**. Two independent stresses: multi-rep segmentation, and a frame rate far from the 30 fps the frame-count thresholds assume.
- **`negative_control_layup`** — deliberately out of scope. Also has a pathological container frame rate (1000 fps, millisecond timebase), which tests whether the pipeline validates its inputs.

## Re-download

```bash
mkdir -p assets/videos/external

venv/Scripts/yt-dlp.exe --no-warnings -f "18/mp4[height<=480]/best[ext=mp4]" \
  -o "assets/videos/external/external_01_free_throw_curry_kerr.%(ext)s" \
  "https://www.youtube.com/watch?v=b2E7kCHkSO0"

venv/Scripts/yt-dlp.exe --no-warnings -f "18/mp4[height<=480]/best[ext=mp4]" \
  -o "assets/videos/external/external_02_jump_shot_peja.%(ext)s" \
  "https://www.youtube.com/watch?v=Bf7NvlRRKY4"

curl -sL -o assets/videos/external/negative_control_layup_van_rossom.webm \
  "https://upload.wikimedia.org/wikipedia/commons/2/20/Safata_Van_Rossom.webm"
```

## Attribution

The Wikimedia file is CC BY-SA 3.0 by **Coentor** and must be attributed if
redistributed. The YouTube files are Creative Commons Attribution; attribute the
original uploaders. Used here for academic evaluation only.
