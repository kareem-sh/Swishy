"""Checkpoint save/load for Swichy ML.

A checkpoint is a Python ``dict`` serialized with ``torch.save``.

Typical keys
------------
- ``model_state_dict``: learnable weights
- ``optimizer_state_dict``: AdamW moments (needed to resume)
- ``scheduler_state_dict``: LR schedule position
- ``epoch``: last finished epoch index
- ``best_metric``: best validation score so far
- ``config``: frozen hyperparameter snapshot
- ``rng``: optional future seed metadata

TODO:
    - Export ONNX from best_model.pt.
    - Add SHA256 of weights for integrity checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from ml.utils.config import get_repo_root


def resolve_checkpoint_dir(config: Mapping[str, Any]) -> Path:
    raw = Path(str(config["checkpoint"]["dir"]))
    path = raw if raw.is_absolute() else (get_repo_root() / raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any | None,
    epoch: int,
    best_metric: float,
    config: Mapping[str, Any],
    scaler: Any | None = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a full training checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": dict(config),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if extra:
        payload.update(dict(extra))

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load weights (and optionally optimizer state) from disk.

    ``map_location`` lets Salah load a CUDA-trained checkpoint on CPU.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
