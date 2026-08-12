"""Main training entrypoint for Swichy ML.

Run from the repo root::

    .\\venv\\Scripts\\Activate.ps1
    python -m ml.training.train

This script wires together config, device, seed, dataset, model, AMP,
checkpoints, TensorBoard, and early stopping.

It does NOT touch the rule engine in ``analysis/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow ``python ml/training/train.py`` as well as ``python -m ml.training.train``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from ml.datasets import make_dataloaders
from ml.evaluation.metrics import classification_scores
from ml.models import build_model_from_config
from ml.training.checkpointing import (
    load_checkpoint,
    load_checkpoint_payload,
    resolve_checkpoint_dir,
    save_checkpoint,
)
from ml.training.engine import evaluate_one_epoch, train_one_epoch
from ml.training.optim import build_optimizer, build_scheduler
from ml.utils.config import load_train_config, summarize_config
from ml.utils.device import (
    collect_device_info_from_config,
    format_device_report,
    format_parameter_report,
)
from ml.utils.seed import set_seed_from_config


def _metric_value(name: str, val_metrics: dict[str, float], f1_macro: float) -> float:
    name = name.lower()
    if name == "accuracy":
        return float(val_metrics["accuracy"])
    if name == "f1_macro":
        return float(f1_macro)
    if name == "loss":
        # Lower is better — early-stopping code inverts comparison for loss.
        return float(val_metrics["loss"])
    raise ValueError(f"Unknown early_stopping_metric: {name}")


def _is_improvement(
    metric_name: str,
    candidate: float,
    best: float,
) -> bool:
    if metric_name.lower() == "loss":
        return candidate < best
    return candidate > best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swichy shot-quality MLP")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to train.yaml (default: ml/configs/train.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_train_config(args.config)

    print("=== Swichy ML training ===")
    print(summarize_config(config))
    print("---")

    set_seed_from_config(config)
    info = collect_device_info_from_config(config)
    print(format_device_report(info))
    device = info.device

    ckpt_dir = resolve_checkpoint_dir(config)
    best_path = ckpt_dir / str(config["checkpoint"]["best_filename"])
    last_path = ckpt_dir / str(config["checkpoint"]["last_filename"])
    resume_requested = bool(config["checkpoint"].get("resume", False))
    resume_preview = (
        load_checkpoint_payload(last_path, map_location="cpu")
        if resume_requested and last_path.is_file()
        else None
    )
    saved_normalization = (
        resume_preview.get("feature_normalization")
        if resume_preview is not None
        else None
    )

    train_loader, val_loader, train_ds, _ = make_dataloaders(
        config,
        normalization_stats=saved_normalization,
    )
    model = build_model_from_config(config).to(device)
    print(format_parameter_report(model))
    print(model)

    num_classes = int(config["data"]["num_classes"])
    weighting = str(config["training"].get("class_weighting", "none")).lower()
    if weighting == "balanced":
        counts = torch.bincount(train_ds.labels, minlength=num_classes).float()
        weights = counts.sum() / (num_classes * counts)
        weights = weights.to(device)
        print(f"Class counts:  {[int(value) for value in counts.tolist()]}")
        print(f"Class weights: {[round(float(value), 4) for value in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif weighting in {"none", "off", ""}:
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("training.class_weighting must be balanced or none")
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_name = build_scheduler(optimizer, config)

    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    early_metric_name = str(config["training"]["early_stopping_metric"])
    best_metric = float("inf") if early_metric_name == "loss" else float("-inf")
    epochs_without_improve = 0

    if resume_requested and last_path.is_file():
        print(f"Resuming from {last_path}")
        ckpt = load_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_metric = float(ckpt.get("best_metric", best_metric))

    tb_dir = Path(str(config["logging"]["tensorboard_dir"]))
    if not tb_dir.is_absolute():
        tb_dir = _REPO_ROOT / tb_dir
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(
        log_dir=str(tb_dir / str(config["experiment"]["name"])),
        flush_secs=int(config["logging"].get("flush_secs", 10)),
    )

    max_epochs = int(config["training"]["epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    grad_clip = float(config["training"]["gradient_clip"])
    log_every = int(config["training"].get("log_every_n_steps", 10))
    normalization_stats = getattr(train_loader, "normalization_stats", None)

    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            gradient_clip=grad_clip,
            mixed_precision=use_amp,
            log_every_n_steps=log_every,
        )
        val_metrics, val_logits, val_labels = evaluate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            mixed_precision=use_amp,
        )
        val_preds = torch.argmax(val_logits, dim=1)
        scores = classification_scores(val_labels, val_preds, num_classes)
        f1_macro = float(scores["f1_macro"])

        candidate = _metric_value(early_metric_name, val_metrics, f1_macro)

        if scheduler is not None:
            if scheduler_name == "plateau":
                scheduler.step(candidate)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} "
            f"f1_macro={f1_macro:.3f} | lr={lr:.6f}"
        )

        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("f1/val_macro", f1_macro, epoch)
        writer.add_scalar("lr", lr, epoch)

        improved = _is_improvement(early_metric_name, candidate, best_metric)

        if bool(config["checkpoint"].get("save_last", True)):
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric if not improved else candidate,
                config=config,
                scaler=scaler,
                extra={"feature_normalization": normalization_stats},
            )

        if improved:
            best_metric = candidate
            epochs_without_improve = 0
            if bool(config["checkpoint"].get("save_best", True)):
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    scaler=scaler,
                    extra={"feature_normalization": normalization_stats},
                )
                print(f"  saved best checkpoint -> {best_path}")
        else:
            epochs_without_improve += 1
            print(
                f"  no improvement ({epochs_without_improve}/{patience}) "
                f"on {early_metric_name}"
            )
            if epochs_without_improve >= patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print("\nTraining finished.")
    print(f"Best metric ({early_metric_name}) = {best_metric:.4f}")
    print(f"Best model: {best_path}")
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
