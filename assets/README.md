# Test Assets

Free media for testing Swichy pose analysis. See also [MANUAL_COMPLETION_GUIDE.md](../docs/MANUAL_COMPLETION_GUIDE.md) for which clip to use when studying each feature.

---

## Videos (`assets/videos/`)

| File | Source | Shot type / camera | Notes |
|------|--------|-------------------|-------|
| [`test.mp4`](test.mp4) | [Mixkit #2280](https://mixkit.co/free-stock-video/couple-playing-basketball-2280/) | **Front-view jump shot** | Default video (~11s slow-mo outdoor three-pointer) |
| [`video_01_free_throw.mp4`](videos/video_01_free_throw.mp4) | [Mixkit #2278](https://mixkit.co/free-stock-video/basketball-player-shot-at-goal-2278/) | Free throw (front) | Outdoor free-throw practice (~14s) |
| [`video_02_one_on_one.mp4`](videos/video_02_one_on_one.mp4) | [Mixkit #44469](https://mixkit.co/free-stock-video/two-basketball-players-playing-one-on-one-44469/) | **Hook / layup** | Indoor one-on-one (~8s) |
| [`video_03_expert_score.mp4`](videos/video_03_expert_score.mp4) | [Mixkit #44449](https://mixkit.co/free-stock-video/expert-basketball-player-scoring-a-basket-44449/) | **Dunk** | Indoor scoring move (~6s) |
| [`video_04_shooting_alone.mp4`](videos/video_04_shooting_alone.mp4) | [Mixkit #44448](https://mixkit.co/free-stock-video/skilled-basketball-player-shooting-baskets-training-alone-44448/) | **Dunk** | Solo drills clip (~6s) |
| [`video_05_pair_training.mp4`](videos/video_05_pair_training.mp4) | [Mixkit #44460](https://mixkit.co/free-stock-video/pair-of-basketball-players-training-shots-44460/) | **Back-view jump shot** | Two players training (~10s) |
| [`video_06_dunk.mp4`](videos/video_06_dunk.mp4) | [Mixkit #2283](https://mixkit.co/free-stock-video/basketball-player-jumping-to-score-2283/) | **Dunk** | Outdoor jump to rim (~8s) |
| [`video_07_side_jump_shot.mp4`](videos/video_07_side_jump_shot.mp4) | [YouTube](https://www.youtube.com/watch?v=furio3awLGY) | **Side-view jump shot** | Best clip for sagittal biomechanics analysis |

**Licenses:** Mixkit clips use the [Mixkit Free License](https://mixkit.co/license/). The YouTube clip is for local testing only — check the video's YouTube license/terms before redistribution.

---

## Recommended clips by use case

| Goal | Use this file |
|------|----------------|
| Front-view jump shot | `test.mp4` |
| Side-view jump shot (biomechanics) | `video_07_side_jump_shot.mp4` |
| Back-view jump shot | `video_05_pair_training.mp4` |
| Dunk / rim finishes | `video_03_expert_score.mp4`, `video_04_shooting_alone.mp4`, `video_06_dunk.mp4` |
| Hook / layup | `video_02_one_on_one.mp4` |

---

## Images (`assets/images/`)

| File | Source | Description |
|------|--------|-------------|
| [`image_01_durant_jumpshot.jpg`](images/image_01_durant_jumpshot.jpg) | [Wikimedia — Kevin Durant jumpshot](https://commons.wikimedia.org/wiki/File:Kevin_Durant_jumpshot.jpg) | Game jump shot, full body |
| [`image_02_hardaway_jumpshot.jpg`](images/image_02_hardaway_jumpshot.jpg) | [Wikimedia — Tim Hardaway Jr](https://commons.wikimedia.org/wiki/File:Tim_Hardaway_Jr_jump_shot_Final_Four_2013.jpg) | NCAA Final Four jump shot |
| [`image_03_basketball_shoot.jpg`](images/image_03_basketball_shoot.jpg) | [Wikimedia — Basketball shoot 1](https://commons.wikimedia.org/wiki/File:Basketball_shoot_1.jpg) | Player shooting |
| [`image_04_vt_jump_shot.jpg`](images/image_04_vt_jump_shot.jpg) | [Wikimedia — VT vs Robert Morris](https://commons.wikimedia.org/wiki/File:2013_Virginia_Tech_-_Robert_Morris_-_jump_shot.jpg) | College game jump shot |
| [`image_05_brackins_jumpshot.jpg`](images/image_05_brackins_jumpshot.jpg) | [Wikimedia — Craig Brackins](https://commons.wikimedia.org/wiki/File:Craig_Brackins_jumpshot.jpg) | Jump shot form |

**License:** Creative Commons (see each file page on Wikimedia Commons).

Default image for image mode: [`test.jpg`](test.jpg) (copy of `image_01_durant_jumpshot.jpg`).

---

## How to run

### Video mode

In [`main.py`](../main.py):

```python
MODE = "video"
```

Examples:

```python
# Side-view jump shot (best for biomechanics)
run_video_mode("assets/videos/video_07_side_jump_shot.mp4")

# Front-view jump shot (default)
run_video_mode("assets/test.mp4")
```

### Image mode

```python
MODE = "image"
run_image_mode("assets/images/image_02_hardaway_jumpshot.jpg")
```

---

## Re-download everything

```powershell
# Mixkit videos
$videos = @{
    "assets\test.mp4" = "https://assets.mixkit.co/videos/2280/2280-720.mp4"
    "assets\videos\video_01_free_throw.mp4" = "https://assets.mixkit.co/videos/2278/2278-720.mp4"
    "assets\videos\video_02_one_on_one.mp4" = "https://assets.mixkit.co/videos/44469/44469-720.mp4"
    "assets\videos\video_03_expert_score.mp4" = "https://assets.mixkit.co/videos/44449/44449-720.mp4"
    "assets\videos\video_04_shooting_alone.mp4" = "https://assets.mixkit.co/videos/44448/44448-720.mp4"
    "assets\videos\video_05_pair_training.mp4" = "https://assets.mixkit.co/videos/44460/44460-720.mp4"
    "assets\videos\video_06_dunk.mp4" = "https://assets.mixkit.co/videos/2283/2283-720.mp4"
}
foreach ($item in $videos.GetEnumerator()) {
    Invoke-WebRequest -Uri $item.Value -OutFile $item.Key -UseBasicParsing
}

# Side-view jump shot (YouTube — requires yt-dlp)
.\venv\Scripts\yt-dlp.exe -f "18/best[ext=mp4][height<=720]/best" `
    -o "assets\videos\video_07_side_jump_shot.%(ext)s" `
    "https://www.youtube.com/watch?v=furio3awLGY"

Copy-Item assets\images\image_01_durant_jumpshot.jpg assets\test.jpg -Force
```
