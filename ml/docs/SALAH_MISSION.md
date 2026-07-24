# Salah Mission — Train Swichy's Shot-Quality Neural Network

**Owner:** Salah (training + video data)  
**Repo partner:** Karim (pipeline + ML scaffolding)  
**Rule:** Do **not** delete or replace the existing rule engine in `analysis/`.  
The MLP trains **beside** it until it clearly wins on real labeled shots.

**System design (rules + ML + optional ball):**  
[FORM_ML_AND_RULES.md](FORM_ML_AND_RULES.md)

---

## Arabic — ابدأ من هنا (Salah)

1. حط فيديوهات التصويب في `ml/datasets/videos/train/`
2. عدّل `ml/datasets/videos/labels.csv` — أهم عمود هو **`class_id` (جودة الفورم)**  
   السلة / in-out **اختياري** (`made` فاضي إذا ما تشوف النتيجة)
3. صدّر الأرقام ثم درّب:

```powershell
cd C:\path\to\Swichy
.\venv\Scripts\Activate.ps1
python -m ml.datasets.build_features_from_videos --labels ml/datasets/videos/labels.csv --output ml/datasets/data/train.npz
python -m ml.training.train
tensorboard --logdir ml/tensorboard
```

4. الموديل **ما يغيّر** قواعد `biomechanics.yaml` — يتعلم جنب الـ rule engine  
5. التفاصيل: [FORM_ML_AND_RULES.md](FORM_ML_AND_RULES.md)

---

## 1. Mission Overview

You will:

1. Put jump-shot videos under `ml/datasets/videos/train/` (hoop **not** required).
2. Label **form quality** in `labels.csv` (`0` acceptable/correct, `1` needs improvement). Optional: `made`, `has_hoop`.
3. Export features → `train.npz`.
4. Train the form MLP (`python -m ml.training.train`).
5. Watch TensorBoard; later Karim compares MLP vs rule engine on a holdout set.

Important mental model:

```text
Your videos + form labels (class_id)
        ↓
build_features_from_videos.py   (pose / angles / phases → 33-D vector)
        ↓
train.npz
        ↓
MLP training
        ↓
best_model.pt
        ↓
(later) compare beside rule engine — do not delete analysis/
```

The neural net does **not** train on raw MP4 pixels in v1. It trains on numeric
feature vectors. Make/miss is **not** the training target for form v1.

---

## 2. Project Structure

```text
ml/
├── configs/train.yaml          # ALL hyperparameters (source: npy for real form data)
├── datasets/
│   ├── feature_dataset.py
│   ├── build_features_from_videos.py   # videos + labels -> train.npz
│   ├── videos/
│   │   ├── train/              # DROP MP4s HERE
│   │   ├── val/
│   │   ├── labels.csv
│   │   └── labels.example.csv
│   └── data/                   # train.npz (gitignored)
├── models/mlp.py
├── training/train.py
├── evaluation/evaluate.py
├── inference/predict.py
├── checkpoints/                # best_model.pt
├── tensorboard/
└── docs/
    ├── SALAH_MISSION.md        # this file
    └── FORM_ML_AND_RULES.md    # how ML works with the rule engine
```

Live app (unchanged rule path):

```text
pose → filter → features → analysis/BiomechanicsEngine → score
```

---

## 3. Installation

### Python version

Use **Python 3.10+** (Karim verified 3.13.5). Prefer a dedicated venv.

### Create virtual environment (Windows PowerShell)

```powershell
cd C:\path\to\Swichy
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install requirements + CUDA PyTorch

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install tensorboard pyyaml
```

CPU-only fallback (slower):

```powershell
python -m pip install torch torchvision
python -m pip install -r requirements.txt tensorboard pyyaml
```

---

## 4. CUDA Verification

```powershell
python ml\tests\smoke_device.py
python scripts\check_environment.py --require-cuda
```

---

## 5. Optional smoke test (synthetic — not basketball)

Only if you need to verify GPU/trainer without videos. Real form training uses `npy`.

```powershell
# Temporarily set data.source: synthetic in train.yaml, then:
python -m ml.training.train
```

Prefer the real path in section 6.

---

## 6. Where to put videos + labels

```text
ml/datasets/videos/
├── labels.csv          ← copy from labels.example.csv and edit
├── train/
│   └── *.mp4
└── val/
    └── *.mp4
```

See [`ml/datasets/videos/README.md`](../datasets/videos/README.md).

Absolute paths on an external drive are fine in `labels.csv`.

### Form-first label columns

| Column | Required? | Meaning |
|--------|-----------|---------|
| `video_path` | yes | Path to MP4 |
| `shot_index` | yes | `0`, `1`, … or `*` = all shots in file |
| `class_id` | yes | Binary form quality: 0 or 1 |
| `made` | no | `1` / `0` if you can see make/miss |
| `has_hoop` | no | `1` / `0` |
| `notes` | no | Free text |

| class_id | meaning |
|----------|---------|
| 0 | acceptable/correct form |
| 1 | form needs improvement |

---

## 6b. Train on your videos (exact order)

### Step 1 — Videos + labels

1. Copy MP4s into `ml/datasets/videos/train/` (and optionally `val/`).
2. Edit `labels.csv` (start from `labels.example.csv`).

### Step 2 — Export features

```powershell
python -m ml.datasets.build_features_from_videos `
  --videos ml/datasets/videos `
  --labels ml/datasets/videos/labels.csv `
  --output ml/datasets/data/train.npz
```

Optional ball/rim side stats in meta only (does not change form labels):

```powershell
python -m ml.datasets.build_features_from_videos `
  --labels ml/datasets/videos/labels.csv `
  --output ml/datasets/data/train.npz `
  --with-ball
```

### Step 3 — Config

`ml/configs/train.yaml` is already set for form training:

```yaml
data:
  source: npy
  train_path: ml/datasets/data/train.npz
  feature_dim: 33
  num_classes: 2
model:
  input_size: 33
  output_size: 2
```

After export, confirm `feature_dim` matches the printed `feature_dim` (should be 33).

### Step 4 — Train + evaluate

```powershell
python -m ml.training.train
python -m ml.evaluation.evaluate
tensorboard --logdir ml\tensorboard
```

Open `http://localhost:6006/` → Scalars → loss / accuracy.

---

## 7. Hyperparameters

See `ml/configs/train.yaml`. On GTX 1650, small `batch_size` is fine for this MLP.
Windows: keep `num_workers: 0`.

---

## 8. Git Workflow

```powershell
git pull
git checkout -b ml/salah-form-data
# do NOT commit large .mp4 or train.npz
git add ml/docs ml/configs ml/datasets/*.py
git status
git commit -m "docs(ml): form-first training notes"
```

**Do not commit:** raw video folders, huge `.mp4`, multi-GB datasets.  
**Do commit:** code, docs, `train.yaml`.

---

## 9. Common Errors

| Error | Fix |
|-------|-----|
| `model.input_size must equal data.feature_dim` | Match both to export `feature_dim` (33) |
| Checkpoint not found | Run training first |
| CUDA OOM | Lower `batch_size` |
| Windows DataLoader hang | Keep `num_workers: 0` |
| BatchNorm size-1 error | Exporter/train use `drop_last` when needed |
| Loss is NaN | Check features for NaN from pose gaps |
| Accuracy stuck at chance | Too few labeled shots; keep collecting |

---

## 10. Future Improvements (ordered)

1. Collect hundreds → thousands of **form** labels (priority).
2. Compare MLP vs rule score on a fair holdout ([FORM_ML_AND_RULES.md](FORM_ML_AND_RULES.md)).
3. Wire MLP into `pipeline.py` **in parallel** (not a silent replacement).
4. Optional: multi-task make/miss when enough hoop videos exist.
5. Optional: calibrate `biomechanics.yaml` thresholds (deferred).
6. Export ONNX for mobile.

---

## 11. Checklist — Salah Tasks

- [ ] Clone/pull Swichy and create venv  
- [ ] Install CUDA PyTorch + requirements + tensorboard  
- [ ] Read [FORM_ML_AND_RULES.md](FORM_ML_AND_RULES.md)  
- [ ] Put videos in `ml/datasets/videos/train/`  
- [ ] Fill `labels.csv` (`class_id` required; `made` optional)  
- [ ] Export `train.npz`  
- [ ] Run `python -m ml.training.train`  
- [ ] Open TensorBoard  
- [ ] Share metrics + `best_model.pt` with Karim for pipeline comparison  

---

## 12. Contact Split

| Person | Owns |
|--------|------|
| **Salah** | Videos, form labels, feature export, real training runs, metrics |
| **Karim** | Live pipeline, rule engine, ball/rim, ML scaffolding, later MLP↔rules fusion |

When your `best_model.pt` beats the rule engine on a fair test set, Karim wires
it into `pipeline.py` **in parallel** (not as a silent replacement).
