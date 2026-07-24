"""Run inference with a trained Swichy ML checkpoint.

Examples
--------
Single random synthetic vector (smoke)::

    python -m ml.inference.predict --demo

From a CSV / NPY feature file (Salah, after export)::

    python -m ml.inference.predict --features path/to/features.npy

Concepts
--------
1. Load weights onto ``device``.
2. ``model.eval()`` disables Dropout randomness and freezes BatchNorm stats.
3. Forward → logits shape ``(B, C)``.
4. Softmax → probabilities that sum to 1 along classes.
5. Argmax → predicted class index.
6. Confidence = probability of the chosen class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from ml.datasets.feature_dataset import normalize_features
from ml.models import ShotQualityMLP, build_model_from_config
from ml.training.checkpointing import load_checkpoint
from ml.utils.config import get_repo_root, load_train_config
from ml.utils.device import collect_device_info_from_config, format_device_report


def load_model_for_inference(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[ShotQualityMLP, torch.device, float]:
    """Build model, load best weights, return ``(model, device, temperature)``."""
    if device is None:
        info = collect_device_info_from_config(config)
        device = info.device

    model = build_model_from_config(config).to(device)
    ckpt = checkpoint_path or config["inference"]["checkpoint_path"]
    path = Path(ckpt)
    if not path.is_absolute():
        path = get_repo_root() / path
    checkpoint = load_checkpoint(path, model=model, map_location=device)
    model.feature_normalization = checkpoint.get("feature_normalization")
    model.eval()
    temperature = float(config["inference"].get("temperature", 1.0))
    if temperature <= 0:
        raise ValueError("inference.temperature must be > 0")
    return model, device, temperature


@torch.no_grad()
def predict_proba(
    model: ShotQualityMLP,
    features: torch.Tensor | np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return class probabilities.

    Parameters
    ----------
    features:
        Shape ``(F,)`` or ``(B, F)``.
    temperature:
        Softmax temperature. 1.0 = normal. >1 softer, <1 sharper.

    Returns
    -------
    probs:
        Shape ``(B, C)`` on CPU.
    """
    tensor = torch.as_tensor(features, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)  # (F,) → (1, F)
    if tensor.ndim != 2:
        raise ValueError(f"features must be (F,) or (B, F), got {tuple(tensor.shape)}")

    normalization = getattr(model, "feature_normalization", None)
    if normalization is not None:
        tensor = normalize_features(tensor, normalization)

    tensor = tensor.to(device)
    logits = model(tensor) / temperature
    probs = F.softmax(logits, dim=1)
    return probs.cpu()


@torch.no_grad()
def predict_classes(
    model: ShotQualityMLP,
    features: torch.Tensor | np.ndarray,
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pred_class, confidence)`` each shape ``(B,)``."""
    probs = predict_proba(model, features, device, temperature=temperature)
    conf, pred = torch.max(probs, dim=1)
    return pred, conf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swichy ML inference")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on one random synthetic feature vector",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to .npy feature matrix (N, F) or vector (F,)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_train_config(args.config)
    info = collect_device_info_from_config(config)
    print(format_device_report(info))

    model, device, temperature = load_model_for_inference(
        config,
        checkpoint_path=args.checkpoint,
        device=info.device,
    )
    print(f"Model ready on {device} (temperature={temperature})")

    if args.demo:
        rng = np.random.default_rng(0)
        features = rng.normal(
            size=(int(config["model"]["input_size"]),)
        ).astype(np.float32)
        print(f"demo features shape: {features.shape}")
    elif args.features:
        path = Path(args.features)
        if not path.is_absolute():
            path = get_repo_root() / path
        features = np.load(path).astype(np.float32)
        print(f"loaded features shape: {features.shape}")
    else:
        print("Provide --demo or --features path")
        return 2

    pred, conf = predict_classes(model, features, device, temperature=temperature)
    probs = predict_proba(model, features, device, temperature=temperature)

    for i in range(pred.shape[0]):
        print(
            f"sample[{i}]: class={int(pred[i])} "
            f"confidence={float(conf[i]):.4f} "
            f"probs={probs[i].tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
