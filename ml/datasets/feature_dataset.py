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
    train_ds = FeatureDataset(train_features, train_labels)

    val_path = data_cfg.get("val_path")
    if val_path and str(val_path).strip():
        val_path_resolved = _resolve_data_path(str(val_path))
        if val_path_resolved.exists():
            val_features, val_labels = load_arrays_from_path(str(val_path), source=source)
            return train_ds, FeatureDataset(val_features, val_labels)

    return train_ds, None


def make_dataloaders(
    config: Mapping[str, Any],
) -> Tuple[DataLoader, DataLoader, FeatureDataset, FeatureDataset]:
    """Create train/val DataLoaders according to ``train.yaml``.

    If no validation file exists, splits the training set with
    ``data.validation_split``.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    seed = int(config.get("reproducibility", {}).get("random_seed", 42))

    train_ds, val_ds = load_feature_dataset(config)

    if val_ds is None:
        split = float(data_cfg["validation_split"])
        n_total = len(train_ds)
        n_val = max(1, int(round(n_total * split)))
        n_train = n_total - n_val
        if n_train < 1:
            raise ValueError("validation_split left zero training samples")
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(
            train_ds,
            lengths=[n_train, n_val],
            generator=generator,
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
    return train_loader, val_loader, train_ds, val_ds
