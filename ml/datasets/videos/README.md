# Where to put training videos (Salah)

**Form first.** You only need player jump-shot videos + a form grade.  
Hoop / make-miss is optional. Full design: [`FORM_ML_AND_RULES.md`](../docs/FORM_ML_AND_RULES.md).

## Recommended layout

```text
ml/datasets/videos/
├── README.md          ← this file
├── labels.csv         ← required: form class_id per shot (or * for all shots)
├── labels.example.csv
├── train/             ← DROP MOST CLIPS HERE
│   ├── player_a_01.mp4
│   └── ...
└── val/               ← holdout clips (optional but better)
    └── ...
```

## labels.csv format (form-first)

```csv
video_path,shot_index,class_id,made,has_hoop,notes
train/player_a_01.mp4,0,0,1,1,excellent form and made
train/player_a_01.mp4,1,3,,0,poor release — no hoop in frame
train/player_a_02.mp4,*,1,,,all shots in this clip share class 1
assets/videos/video_07_side_jump_shot.mp4,*,1,,,repo path also works
```

| Column | Required | Meaning |
|--------|----------|---------|
| `video_path` | yes | Relative to this folder, repo root, or absolute |
| `shot_index` | yes | `0` = first shot, or `*` / `-1` = every shot |
| `class_id` | yes | Form: 0 excellent … 4 major error |
| `made` | no | `1` / `0` / blank if unknown |
| `has_hoop` | no | `1` / `0` / blank |
| `notes` | no | Free text |

Blank `made` / `has_hoop` is fine — form training still runs.

## Export then train

```powershell
python -m ml.datasets.build_features_from_videos `
  --videos ml/datasets/videos `
  --labels ml/datasets/videos/labels.csv `
  --output ml/datasets/data/train.npz

# Optional: ball/rim side metrics in .meta.json only
#   ... --with-ball

python -m ml.training.train
tensorboard --logdir ml/tensorboard
```

`ml/configs/train.yaml` already uses `source: npy` and `feature_dim: 33`.

## Important

1. Videos are **not** fed straight into the MLP.
2. Training target is **form** (`class_id`), not make/miss.
3. Do **not** commit large `.mp4` files to git.
4. The rule engine in `analysis/` is **not** modified by training.
