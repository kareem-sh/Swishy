"""Synthetic basketball-like feature arrays for pipeline smoke tests.

This is NOT real basketball data. It only verifies:

  Dataset → DataLoader → model → loss → backward → checkpoint

Salah should never train a production model on this file's output.

TODO:
    - Delete reliance on synthetic once real .npy exports exist.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def make_synthetic_arrays(
    num_samples: int,
    feature_dim: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create reproducible fake features ``X`` and integer labels ``y``.

    Shapes
    ------
    X : (num_samples, feature_dim) float32
    y : (num_samples,) int64 in ``{0, ..., num_classes - 1}``

    How it works
    ------------
    1. Draw random Gaussian features.
    2. Build a fixed random linear projection ``W`` of shape
       ``(feature_dim, num_classes)``.
    3. Labels = argmax(X @ W). So there *is* a learnable pattern — the MLP
       can drive training loss down (proves backprop works).
    """
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(num_samples, feature_dim))
    features = features.astype(np.float32)

    weights = rng.normal(size=(feature_dim, num_classes)).astype(np.float32)
    logits = features @ weights
    labels = np.argmax(logits, axis=1).astype(np.int64)
    return features, labels


class SyntheticShotDataset(Dataset):
    """PyTorch Dataset wrapping synthetic ``(features, label)`` pairs.

    Parameters
    ----------
    features:
        Array shape ``(N, F)``.
    labels:
        Array shape ``(N,)``.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
        if len(features) != len(labels):
            raise ValueError(
                f"Length mismatch: {len(features)} features vs {len(labels)} labels"
            )

        # Store as tensors so ``__getitem__`` does not re-convert every time.
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        """Number of samples — required by DataLoader."""
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one sample.

        Returns
        -------
        features : Tensor shape ``(F,)``
        label : Tensor scalar ``()`` (0-D long)
        """
        return self.features[index], self.labels[index]
