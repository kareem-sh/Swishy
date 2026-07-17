"""Feature-vector datasets for Swichy ML.

Form-first flow::

    videos (Salah) + labels.csv (class_id)
        -> build_features_from_videos.py
        -> train.npz
        -> FeatureDataset / training

Design: ``ml/docs/FORM_ML_AND_RULES.md``. Synthetic data remains only for
trainer smoke tests (``data.source: synthetic``).
"""

from __future__ import annotations

from ml.datasets.feature_dataset import (
    FeatureDataset,
    load_feature_dataset,
    make_dataloaders,
)
from ml.datasets.synthetic import SyntheticShotDataset, make_synthetic_arrays

__all__ = [
    "FeatureDataset",
    "SyntheticShotDataset",
    "load_feature_dataset",
    "make_dataloaders",
    "make_synthetic_arrays",
]
