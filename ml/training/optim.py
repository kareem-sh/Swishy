"""Build optimizer and LR scheduler from train.yaml."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LRScheduler,
)


def build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> Optimizer:
    opt_cfg = config["optimizer"]
    name = str(opt_cfg["name"]).lower()
    lr = float(opt_cfg["learning_rate"])
    weight_decay = float(opt_cfg["weight_decay"])

    if name != "adamw":
        raise ValueError(f"Unsupported optimizer '{name}'. v1 supports adamw only.")

    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))
    return AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: Mapping[str, Any],
) -> Tuple[LRScheduler | ReduceLROnPlateau | None, str]:
    sch_cfg = config["scheduler"]
    name = str(sch_cfg.get("name", "none")).lower()
    epochs = int(config["training"]["epochs"])

    if name in {"none", "null", ""}:
        return None, "none"
    if name == "cosine":
        return (
            CosineAnnealingLR(
                optimizer,
                T_max=max(epochs, 1),
                eta_min=float(sch_cfg.get("eta_min", 1e-5)),
            ),
            "cosine",
        )
    if name == "step":
        return (
            StepLR(
                optimizer,
                step_size=int(sch_cfg.get("step_size", 10)),
                gamma=float(sch_cfg.get("gamma", 0.1)),
            ),
            "step",
        )
    if name == "plateau":
        metric_name = str(config["training"].get("early_stopping_metric", "accuracy")).lower()
        return (
            ReduceLROnPlateau(
                optimizer,
                mode="min" if metric_name == "loss" else "max",
                factor=float(sch_cfg.get("plateau_factor", 0.5)),
                patience=int(sch_cfg.get("plateau_patience", 3)),
            ),
            "plateau",
        )
    raise ValueError(f"Unsupported scheduler '{name}'")
