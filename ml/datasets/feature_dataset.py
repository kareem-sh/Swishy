"""Reusable feature Dataset that loads CSV, NPY, or synthetic arrays.

WHY THIS FILE EXISTS
--------------------
Training code should not care *where* features came from.
Only this module knows about:

  - synthetic generation
  - ``.csv`` exports
  - ``.npy`` exports (preferred for large arrays)

When Salah finishes MediaPipe export, he changes ``data.source`` in YAML
and points ``train_path`` / ``val_path`` at his files. ``train.py`` stays
unchanged.

TODO:
    - Replace synthetic dataset with MediaPipe feature dataset.
    - Add optional feature normalization (mean/std from train split only).
    - Support a directory of per-shot ``.npz`` files.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ml.datasets.synthetic import SyntheticShotDataset, make_synthetic_arrays
from ml.utils.config import get_repo_root


class FeatureDataset(Dataset):
    """Generic ``(feature_vector, class_label)`` dataset.

    Parameters
    ----------
    features:
        ``(N, F)`` float array / tensor.
    labels:
        ``(N,)`` integer class indices.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        groups: Sequence[str] | None = None,
    ) -> None:
        features_t = torch.as_tensor(features, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        if features_t.ndim != 2:
            raise ValueError(f"features must be (N, F), got {tuple(features_t.shape)}")
        if labels_t.ndim != 1:
            raise ValueError(f"labels must be (N,), got {tuple(labels_t.shape)}")
        if features_t.shape[0] != labels_t.shape[0]:
            raise ValueError("features and labels must have the same length")

        self.features = features_t
        self.labels = labels_t
        if groups is not None and len(groups) != features_t.shape[0]:
            raise ValueError("groups must have one entry per feature row")
        self.groups = None if groups is None else tuple(str(group) for group in groups)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return int(torch.max(self.labels).item()) + 1


def _resolve_data_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (get_repo_root() / path).resolve()


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV where the last column is the integer label."""
    # genfromtxt keeps this dependency-light (no pandas required).
    array = np.genfromtxt(path, delimiter=",", dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] < 2:
        raise ValueError(f"CSV must have features + label columns: {path}")
    features = array[:, :-1].astype(np.float32)
    labels = array[:, -1].astype(np.int64)
    return features, labels


def _load_npy_pair(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load features/labels from ``.npy`` or ``.npz`` sidecar files.

    Accepted layouts
    ----------------
    1. ``path`` is ``train.npy`` containing a dict-like npz? — prefer explicit pairs:
       ``train_features.npy`` + ``train_labels.npy``
    2. If ``path`` ends with ``.npz``, expects keys ``features`` and ``labels``.
    3. If ``path`` is ``something.csv`` sibling pattern — not used here.
    """
    if path.suffix == ".npz":
        data = np.load(path)
        if "features" not in data or "labels" not in data:
            raise ValueError(
                f"{path} must contain arrays named 'features' and 'labels'"
            )
        return data["features"].astype(np.float32), data["labels"].astype(np.int64)

    # If user pointed at ``train_features.npy``, look for ``train_labels.npy``.
    if path.name.endswith("_features.npy"):
        labels_path = path.with_name(path.name.replace("_features.npy", "_labels.npy"))
        features = np.load(path).astype(np.float32)
        labels = np.load(labels_path).astype(np.int64)
        return features, labels

    # Single ``.npy`` is ambiguous — require .npz or _features/_labels.
    raise ValueError(
        f"For npy source, use a .npz with keys features/labels, "
        f"or files named *_features.npy and *_labels.npy. Got: {path}"
    )


def _load_video_groups(path_str: str, expected_rows: int) -> list[str] | None:
    """Load per-shot video identities from the exporter's ``.meta.json``.

    Group identities let the automatic validation split keep every shot from
    one source video on the same side of the train/validation boundary.
    """
    path = _resolve_data_path(path_str)
    if path.suffix != ".npz":
        return None
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        shots = payload["shots"]
        groups = [str(shot["video"]) for shot in shots]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset metadata for grouped split: {meta_path}") from exc
    if len(groups) != expected_rows:
        raise ValueError(
            f"Metadata row count ({len(groups)}) does not match feature rows "
            f"({expected_rows}): {meta_path}. Re-export the dataset."
        )
    return groups


def fit_feature_normalization(
    features: np.ndarray | torch.Tensor,
) -> dict[str, list[float]]:
    """Fit per-feature mean/std using training rows only."""
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] < 1:
        raise ValueError("Cannot fit normalization without a non-empty (N, F) matrix")
    mean = tensor.mean(dim=0)
    std = tensor.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def normalize_features(
    features: np.ndarray | torch.Tensor,
    stats: Mapping[str, Any],
) -> torch.Tensor:
    """Apply saved training-set normalization to a feature matrix."""
    tensor = torch.as_tensor(features, dtype=torch.float32)
    mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["std"], dtype=torch.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError("Normalization mean/std must be one-dimensional and equal-sized")
    if tensor.shape[-1] != mean.numel():
        raise ValueError(
            f"Normalization expects {mean.numel()} features, got {tensor.shape[-1]}"
        )
    return (tensor - mean) / std


def _subset_dataset(dataset: FeatureDataset, indices: Sequence[int]) -> FeatureDataset:
    index_tensor = torch.as_tensor(list(indices), dtype=torch.long)
    groups = None
    if dataset.groups is not None:
        groups = [dataset.groups[index] for index in indices]
    return FeatureDataset(
        dataset.features[index_tensor],
        dataset.labels[index_tensor],
        groups=groups,
    )


def _class_counts(dataset: FeatureDataset, num_classes: int) -> torch.Tensor:
    labels = dataset.labels
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(
            f"Dataset labels must be in [0, {num_classes - 1}], got "
            f"min={int(labels.min())}, max={int(labels.max())}."
        )
    return torch.bincount(labels, minlength=num_classes)


def _require_class_coverage(
    dataset: FeatureDataset,
    *,
    num_classes: int,
    minimum: int,
    split_name: str,
) -> None:
    counts = _class_counts(dataset, num_classes)
    missing = [idx for idx, count in enumerate(counts.tolist()) if count < minimum]
    if missing:
        rendered = {idx: int(count) for idx, count in enumerate(counts.tolist())}
        raise ValueError(
            f"{split_name} does not have enough examples for classes {missing}. "
            f"Counts={rendered}; need at least {minimum} per class. "
            "Collect/relabel data before training."
        )


def _grouped_split_indices(
    dataset: FeatureDataset,
    *,
    validation_split: float,
    seed: int,
    num_classes: int,
) -> tuple[list[int], list[int]]:
    """Choose a deterministic, approximately stratified split by source video."""
    if dataset.groups is None:
        raise ValueError(
            "data.split_strategy=group_by_video requires the train.meta.json "
            "written beside train.npz. Re-export the dataset."
        )

    unique_groups = sorted(set(dataset.groups))
    if len(unique_groups) < 2:
        raise ValueError("Grouped validation needs shots from at least two videos")

    labels = dataset.labels.numpy()
    group_rows = {
        group: np.asarray(
            [idx for idx, value in enumerate(dataset.groups) if value == group],
            dtype=np.int64,
        )
        for group in unique_groups
    }
    groups_per_class = {
        class_id: sum(bool(np.any(labels[rows] == class_id)) for rows in group_rows.values())
        for class_id in range(num_classes)
    }
    insufficient = [class_id for class_id, count in groups_per_class.items() if count < 2]
    if insufficient:
        raise ValueError(
            "A leakage-safe train/validation split needs each class in at least "
            f"two different videos. Classes {insufficient} have video counts "
            f"{groups_per_class}."
        )

    n_groups = len(unique_groups)
    rng = np.random.default_rng(seed)
    if n_groups <= 14:
        candidates = (
            combo
            for size in range(1, n_groups)
            for combo in combinations(range(n_groups), size)
        )
    else:
        masks: set[tuple[int, ...]] = set()
        for _ in range(4096):
            mask = rng.random(n_groups) < validation_split
            if not mask.any() or mask.all():
                continue
            masks.add(tuple(np.flatnonzero(mask).tolist()))
        candidates = iter(masks)

    total_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    target_rows = len(dataset) * validation_split
    target_counts = total_counts * validation_split
    best_score = float("inf")
    best_val: list[int] | None = None

    for candidate in candidates:
        selected_groups = [unique_groups[index] for index in candidate]
        val_indices = np.concatenate([group_rows[group] for group in selected_groups])
        val_counts = np.bincount(labels[val_indices], minlength=num_classes).astype(np.float64)
        train_counts = total_counts - val_counts
        if np.any(val_counts == 0) or np.any(train_counts == 0):
            continue
        size_error = abs(len(val_indices) - target_rows) / max(len(dataset), 1)
        class_error = np.abs(val_counts - target_counts).sum() / max(len(dataset), 1)
        score = size_error + class_error
        if score < best_score:
            best_score = score
            best_val = sorted(int(index) for index in val_indices.tolist())

    if best_val is None:
        raise ValueError(
            "Could not create a video-grouped split containing every class in "
            "both train and validation. Add more labeled videos per class."
        )
    val_set = set(best_val)
    train_indices = [index for index in range(len(dataset)) if index not in val_set]
    return train_indices, best_val


def load_arrays_from_path(path_str: str, source: str) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch CSV / NPY loading."""
    path = _resolve_data_path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Salah must export MediaPipe features before setting source={source}."
        )
    source = source.lower()
    if source == "csv":
        return _load_csv(path)
    if source in {"npy", "mediapipe"}:
        # mediapipe source currently expects pre-exported npy/npz from a
        # future video builder script.
        return _load_npy_pair(path)
    raise ValueError(f"Unsupported on-disk source '{source}'")


def load_feature_dataset(config: Mapping[str, Any]) -> Tuple[FeatureDataset, Optional[FeatureDataset]]:
    """Build train (and optional val) datasets from config.

    Returns
    -------
    train_dataset, val_dataset
        ``val_dataset`` may be ``None`` when the caller should split train.
    """
    data_cfg = config["data"]
    source = str(data_cfg["source"]).lower()
    seed = int(config.get("reproducibility", {}).get("random_seed", 42))

    if source == "synthetic":
        features, labels = make_synthetic_arrays(
            num_samples=int(data_cfg["synthetic_num_samples"]),
            feature_dim=int(data_cfg["feature_dim"]),
            num_classes=int(data_cfg["num_classes"]),
            seed=seed,
        )
        return FeatureDataset(features, labels), None

    train_features, train_labels = load_arrays_from_path(
        str(data_cfg["train_path"]),
        source=source,
    )
    train_groups = _load_video_groups(str(data_cfg["train_path"]), len(train_labels))
    train_ds = FeatureDataset(train_features, train_labels, groups=train_groups)

    val_path = data_cfg.get("val_path")
    if val_path and str(val_path).strip():
        val_path_resolved = _resolve_data_path(str(val_path))
        if val_path_resolved.exists():
            val_features, val_labels = load_arrays_from_path(str(val_path), source=source)
            val_groups = _load_video_groups(str(val_path), len(val_labels))
            return train_ds, FeatureDataset(
                val_features,
                val_labels,
                groups=val_groups,
            )

    return train_ds, None


def make_dataloaders(
    config: Mapping[str, Any],
    *,
    normalization_stats: Mapping[str, Any] | None = None,
) -> Tuple[DataLoader, DataLoader, FeatureDataset, FeatureDataset]:
    """Create train/val DataLoaders according to ``train.yaml``.

    If no validation file exists, splits the training set with
    ``data.validation_split``.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = int(config.get("reproducibility", {}).get("random_seed", 42))

    train_ds, val_ds = load_feature_dataset(config)
    num_classes = int(data_cfg["num_classes"])
    minimum = int(data_cfg.get("min_samples_per_class", 2))

    _require_class_coverage(
        train_ds,
        num_classes=num_classes,
        minimum=minimum if val_ds is None else 1,
        split_name="Dataset" if val_ds is None else "Training dataset",
    )

    if val_ds is None:
        split = float(data_cfg["validation_split"])
        strategy = str(data_cfg.get("split_strategy", "random")).lower()
        if strategy == "group_by_video" and train_ds.groups is not None:
            train_indices, val_indices = _grouped_split_indices(
                train_ds,
                validation_split=split,
                seed=seed,
                num_classes=num_classes,
            )
            original = train_ds
            train_ds = _subset_dataset(original, train_indices)
            val_ds = _subset_dataset(original, val_indices)
        elif strategy in {"random", "group_by_video"}:
            if strategy == "group_by_video":
                print("WARNING: no video metadata found; falling back to random split")
            n_total = len(train_ds)
            n_val = max(1, int(round(n_total * split)))
            n_train = n_total - n_val
            if n_train < 1:
                raise ValueError("validation_split left zero training samples")
            generator = torch.Generator().manual_seed(seed)
            original = train_ds
            train_subset, val_subset = random_split(
                original,
                lengths=[n_train, n_val],
                generator=generator,
            )
            train_ds = _subset_dataset(original, train_subset.indices)
            val_ds = _subset_dataset(original, val_subset.indices)
        else:
            raise ValueError(
                f"Unknown data.split_strategy={strategy!r}; use group_by_video or random"
            )

    _require_class_coverage(
        train_ds,
        num_classes=num_classes,
        minimum=1,
        split_name="Training split",
    )
    _require_class_coverage(
        val_ds,
        num_classes=num_classes,
        minimum=1,
        split_name="Validation split",
    )

    normalize = bool(data_cfg.get("normalize_features", True))
    fitted_stats: Mapping[str, Any] | None = None
    if normalize:
        fitted_stats = (
            normalization_stats
            if normalization_stats is not None
            else fit_feature_normalization(train_ds.features)
        )
        train_ds = FeatureDataset(
            normalize_features(train_ds.features, fitted_stats),
            train_ds.labels,
            groups=train_ds.groups,
        )
        val_ds = FeatureDataset(
            normalize_features(val_ds.features, fitted_stats),
            val_ds.labels,
            groups=val_ds.groups,
        )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))
    # On CUDA, pin_memory speeds host→device copies. Harmless on CPU.
    shuffle_train = bool(data_cfg.get("shuffle_train", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # BatchNorm1d in the MLP needs >1 sample per batch during training.
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Preserve the existing four-value API while making fitted train-only
    # statistics available to training/checkpoint code.
    train_loader.normalization_stats = fitted_stats  # type: ignore[attr-defined]
    val_loader.normalization_stats = fitted_stats  # type: ignore[attr-defined]
    print(
        "Dataset split: "
        f"train={len(train_ds)} labels={_class_counts(train_ds, num_classes).tolist()} | "
        f"val={len(val_ds)} labels={_class_counts(val_ds, num_classes).tolist()}"
    )
    return train_loader, val_loader, train_ds, val_ds
