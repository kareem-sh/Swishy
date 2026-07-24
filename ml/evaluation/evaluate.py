"""Evaluate a trained checkpoint on the validation / holdout split.

Run::

    python -m ml.evaluation.evaluate

Prints accuracy, precision, recall, F1, confusion matrix, classification report.
Saves a text report under ``evaluation.output_dir``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torch import nn

from ml.datasets import make_dataloaders
from ml.evaluation.metrics import classification_scores
from ml.models import build_model_from_config
from ml.training.checkpointing import load_checkpoint
from ml.training.engine import evaluate_one_epoch
from ml.utils.config import get_repo_root, load_train_config
from ml.utils.device import collect_device_info_from_config, format_device_report
from ml.utils.seed import set_seed_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Swichy ML checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override evaluation.checkpoint_path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_train_config(args.config)
    set_seed_from_config(config)

    info = collect_device_info_from_config(config)
    print(format_device_report(info))
    device = info.device

    model = build_model_from_config(config).to(device)

    ckpt = args.checkpoint or config["evaluation"]["checkpoint_path"]
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = get_repo_root() / ckpt_path

    checkpoint = load_checkpoint(ckpt_path, model=model, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")

    _, val_loader, _, _ = make_dataloaders(
        config,
        normalization_stats=checkpoint.get("feature_normalization"),
    )

    criterion = nn.CrossEntropyLoss()
    use_amp = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    metrics, logits, labels = evaluate_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        mixed_precision=use_amp,
    )
    preds = torch.argmax(logits, dim=1)
    num_classes = int(config["data"]["num_classes"])
    scores = classification_scores(labels, preds, num_classes)

    print(f"val loss: {metrics['loss']:.4f}")
    print(scores["report"])
    print("Confusion matrix (rows=true, cols=pred):")
    print(scores["confusion_matrix"])

    out_dir = Path(str(config["evaluation"]["output_dir"]))
    if not out_dir.is_absolute():
        out_dir = get_repo_root() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.txt"
    cm_path = out_dir / "confusion_matrix.npy"
    report_path.write_text(str(scores["report"]), encoding="utf-8")
    np.save(cm_path, scores["confusion_matrix"])
    print(f"Wrote {report_path}")
    print(f"Wrote {cm_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
