# Test Assets

Free media for testing Swichy pose analysis. All clips and images include full-body or upper-body basketball action suitable for MediaPipe Pose.

---

## Videos (`assets/videos/`)

| File | Source | Duration | Description |
|------|--------|----------|-------------|
| [`test.mp4`](../test.mp4) | [Mixkit #2280](https://mixkit.co/free-stock-video/couple-playing-basketball-2280/) | ~11s | Slow-mo outdoor three-pointer (default video) |
| [`video_01_free_throw.mp4`](videos/video_01_free_throw.mp4) | [Mixkit #2278](https://mixkit.co/free-stock-video/basketball-player-shot-at-goal-2278/) | ~14s | Outdoor free-throw practice |
| [`video_02_one_on_one.mp4`](videos/video_02_one_on_one.mp4) | [Mixkit #44469](https://mixkit.co/free-stock-video/two-basketball-players-playing-one-on-one-44469/) | ~8s | Indoor one-on-one |
| [`video_03_expert_score.mp4`](videos/video_03_expert_score.mp4) | [Mixkit #44449](https://mixkit.co/free-stock-video/expert-basketball-player-scoring-a-basket-44449/) | ~6s | Indoor scoring move |
| [`video_04_shooting_alone.mp4`](videos/video_04_shooting_alone.mp4) | [Mixkit #44448](https://mixkit.co/free-stock-video/skilled-basketball-player-shooting-baskets-training-alone-44448/) | ~6s | Solo shooting drills |
| [`video_05_pair_training.mp4`](videos/video_05_pair_training.mp4) | [Mixkit #44460](https://mixkit.co/free-stock-video/pair-of-basketball-players-training-shots-44460/) | ~10s | Two players passing and shooting |

**License:** [Mixkit Free License](https://mixkit.co/license/) — free for personal and commercial use.

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

Default image for image mode: [`test.jpg`](../test.jpg) (copy of `image_01_durant_jumpshot.jpg`).

---

## How to run

### Video mode

In [`main.py`](../main.py):

```python
MODE = "video"
```

Change the path in `run_video_mode(...)` to any clip, e.g.:

```python
run_video_mode("assets/videos/video_01_free_throw.mp4")
```

### Image mode

```python
MODE = "image"
```

```python
run_image_mode("assets/images/image_02_hardaway_jumpshot.jpg")
```

---

## Re-download everything

```powershell
# Videos (Mixkit)
$videos = @{
    "assets\test.mp4" = "https://assets.mixkit.co/videos/2280/2280-720.mp4"
    "assets\videos\video_01_free_throw.mp4" = "https://assets.mixkit.co/videos/2278/2278-720.mp4"
    "assets\videos\video_02_one_on_one.mp4" = "https://assets.mixkit.co/videos/44469/44469-720.mp4"
    "assets\videos\video_03_expert_score.mp4" = "https://assets.mixkit.co/videos/44449/44449-720.mp4"
    "assets\videos\video_04_shooting_alone.mp4" = "https://assets.mixkit.co/videos/44448/44448-720.mp4"
    "assets\videos\video_05_pair_training.mp4" = "https://assets.mixkit.co/videos/44460/44460-720.mp4"
}
foreach ($item in $videos.GetEnumerator()) {
    Invoke-WebRequest -Uri $item.Value -OutFile $item.Key -UseBasicParsing
}

# Images (Wikimedia — download from each file page linked in the table above)
# Example direct URL for image_01:
# Invoke-WebRequest -Uri "https://upload.wikimedia.org/wikipedia/commons/5/52/Kevin_Durant_jumpshot.jpg" -OutFile "assets\images\image_01_durant_jumpshot.jpg"
Copy-Item assets\images\image_01_durant_jumpshot.jpg assets\test.jpg -Force
```
