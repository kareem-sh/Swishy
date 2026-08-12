# GPU and YOLO Setup

This guide explains the two separate pieces used for ball and rim detection:

1. **Ultralytics runtime** — the Python package that can run YOLO26 and older
   Ultralytics models.
2. **Basketball weights** — the trained parameters that know the classes
   `basketball`, `hoop`, `player`, and `referee`.

Installing the latest Ultralytics package does not convert older custom weights
to YOLO26. Swichy currently runs E-BARD's basketball-specific YOLOv8n weights
because a domain-trained model is more useful than a generic newer model.
The planned training base for Swichy's own dataset is `yolo26n.pt`.

## Current hardware

The development machine detected during setup:

- GPU: NVIDIA GeForce GTX 1650 (4 GB)
- Driver: 592.27
- Driver-reported CUDA capability: 13.1
- Recommended PyTorch wheel: CUDA 12.8 (`cu128`)

The NVIDIA driver is backward compatible with the CUDA 12.8 runtime bundled
inside PyTorch. A separate CUDA Toolkit installation is not required for
inference.

## Clean Windows installation

Run from the repository root:

```powershell
.\venv\Scripts\Activate.ps1

python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade ultralytics huggingface_hub
python -m pip install -r requirements.txt
```

The CUDA PyTorch download is large. If a previous download is still running,
stop that terminal with `Ctrl+C` before starting another installation.

## Verify CUDA

```powershell
python scripts/check_environment.py --require-cuda
```

Expected essentials:

```text
available: True
GPU: NVIDIA GeForce GTX 1650
```

`config/ball.yaml` uses `device: auto`, so Swichy selects GPU `0` when CUDA is
available and falls back to CPU without a code change.

## Verified development environment

Verified on the GTX 1650 development machine:

| Component | Verified value |
|-----------|----------------|
| Python | 3.13.5 |
| Ultralytics | 8.4.98 (YOLO26-capable runtime) |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| CUDA available | `True` |
| Selected Swichy device | GPU `0` |
| MediaPipe | 0.10.35 |
| OpenCV | 4.13.0.92 |

Verification results:

- Custom model classes: `basketball`, `hoop`, `player`, `referee`.
- Still image: ball 1/1 and rim 1/1.
- 120-frame detector test: ball 49 frames, rim 45 frames, about 15.7 FPS.
- Integrated pose + ball/rim test: 120 frames, 70 pose frames, 49 ball
  frames, 45 rim frames, 56 tracked snapshots, about 10.7 FPS.

Detector-only FPS and full-pipeline FPS are intentionally reported separately.
The full pipeline also performs MediaPipe pose analysis, phase detection,
biomechanics, rendering, and video encoding.
Detection coverage is not precision or recall; sticky-rim frames and
short-gap ball predictions deliberately increase temporal coverage.

## Verify the custom model

Image smoke test:

```powershell
python scripts/verify_ball_rim.py --source assets/images/image_03_basketball_shoot.jpg --max-frames 1
```

Video smoke test:

```powershell
python scripts/verify_ball_rim.py --source assets/videos/video_03_expert_score.mp4 --max-frames 120
```

Full integrated smoke test:

```powershell
python scripts/verify_pipeline.py --max-frames 120
```

Compare raw coverage, continuity, confidence, and speed at several inference
sizes:

```powershell
python scripts/benchmark_ball_rim.py --max-frames 120
```

Annotated results are written to `outputs/ball_rim_verify/`.
The script prints the selected inference device, detection counts, and FPS.

## Model choice

| Item | Current choice | Reason |
|------|----------------|--------|
| Runtime | Latest stable `ultralytics` | Supports YOLO26 and existing weights |
| Current weights | E-BARD YOLOv8n | Already trained for basketball + hoop |
| Future training base | `yolo26n.pt` | New Swichy phone-camera training |
| Production export | ONNX (later) | Portable deployment after accuracy is validated |

Do not replace the custom weights with generic `yolo26n.pt` for inference.
Generic YOLO26 weights are not trained with a basketball-hoop class. To use
YOLO26 for both ball and rim, fine-tune `yolo26n.pt` on the Roboflow datasets
listed in [PRODUCT_ROADMAP.md](PRODUCT_ROADMAP.md), then validate it against
Swichy's phone footage.

## Configuration

[`config/ball.yaml`](../config/ball.yaml):

- `device: auto` — GPU when available, CPU fallback.
- `imgsz: 704` — matches the resolution used to train E-BARD.
- `fallback_imgsz: 512` — acquires the static rim at the more robust tested
  scale; after lock, inference switches to the training-matched 704px scale.
- `rim_search_retry_interval: 4` — performs a bounded 704px retry every fourth
  pre-lock frame instead of running both scales on every miss.
- `fallback_on_ball_miss: false` — lets the tracker bridge short gaps instead
  of doubling YOLO work whenever the ball is absent.
- `frame_stride: 1` — evaluates every frame so a fast ball is not skipped.
- `sticky_rim: true` — retains the strongest rim detection for a static camera.
- `min_confidence` / `min_confidence_rim` — separate ball and rim thresholds.
- `rim_lock_max_shift_ratio` — prevents a locked static rim from jumping to a
  distant false detection.
- `rim_center_y_fraction` — places the physical rim near the top of a
  hoop-plus-net detection box.

The multi-scale fallback improves robustness but costs inference time. Final
accuracy still requires labeled phone footage and precision/recall or mAP
evaluation. The largest future gain will come from fine-tuning YOLO26 on
representative Swichy footage, not from lowering confidence further.
