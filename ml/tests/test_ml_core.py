"""Independent ML unit tests. Run each file with: python ml/tests/test_*.py"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from ml.datasets import (
    FeatureDataset,
    fit_feature_normalization,
    make_dataloaders,
    make_synthetic_arrays,
    normalize_features,
)
from ml.models import ShotQualityMLP, build_model_from_config
from ml.training.checkpointing import load_checkpoint, save_checkpoint
from ml.training.optim import build_optimizer
from ml.utils.config import load_train_config, validate_train_config
from ml.utils.device import resolve_device
from ml.utils.seed import set_seed


def test_config_loads_and_validates() -> None:
    config = load_train_config()
    assert config["model"]["input_size"] == config["data"]["feature_dim"]
    bad = load_train_config()
    bad["model"]["output_size"] = 99
    try:
        validate_train_config(bad)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_synthetic_dataset_shapes() -> None:
    x, y = make_synthetic_arrays(100, 32, 5, seed=0)
    assert x.shape == (100, 32)
    assert y.shape == (100,)
    ds = FeatureDataset(x, y)
    feat, label = ds[0]
    assert feat.shape == (32,)
    assert label.ndim == 0


def test_forward_and_backward() -> None:
    set_seed(0)
    model = ShotQualityMLP(32, 64, 5, dropout=0.1, num_hidden_layers=2)
    x = torch.randn(8, 32)
    y = torch.randint(0, 5, (8,))
    logits = model(x)
    assert logits.shape == (8, 5)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_optimizer_steps() -> None:
    config = load_train_config()
    model = build_model_from_config(config)
    opt = build_optimizer(model, config)
    x = torch.randn(4, config["model"]["input_size"])
    y = torch.randint(0, config["data"]["num_classes"], (4,))
    before = model.net[0].weight.detach().clone()
    opt.zero_grad()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()
    after = model.net[0].weight.detach()
    assert not torch.allclose(before, after)


def test_checkpoint_roundtrip(tmp_path: Path | None = None) -> None:
    config = load_train_config()
    model = build_model_from_config(config)
    opt = build_optimizer(model, config)
    out = ROOT / "ml" / "checkpoints" / "_test_ckpt.pt"
    save_checkpoint(
        out,
        model=model,
        optimizer=opt,
        scheduler=None,
        epoch=1,
        best_metric=0.5,
        config=config,
        extra={"feature_normalization": {"mean": [0.0], "std": [1.0]}},
    )
    model2 = build_model_from_config(config)
    checkpoint = load_checkpoint(out, model=model2, map_location="cpu")
    assert checkpoint["feature_normalization"]["std"] == [1.0]
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(a, b)
    out.unlink(missing_ok=True)


def test_device_resolution() -> None:
    cpu = resolve_device("cpu")
    assert str(cpu) == "cpu"
    auto = resolve_device("auto")
    assert auto.type in {"cuda", "mps", "cpu"}


def test_inference_softmax() -> None:
    model = ShotQualityMLP(32, 32, 5, dropout=0.0)
    model.eval()
    x = torch.randn(3, 32)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    assert probs.shape == (3, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-5)


def test_feature_normalization_uses_training_statistics() -> None:
    features = torch.tensor([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    stats = fit_feature_normalization(features)
    normalized = normalize_features(features, stats)
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        normalized.std(dim=0, unbiased=False),
        torch.ones(2),
        atol=1e-6,
    )


def test_grouped_split_has_no_video_leakage() -> None:
    config = load_train_config()
    rng = np.random.default_rng(4)
    features = rng.normal(size=(16, 33)).astype(np.float32)
    labels = np.asarray([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4, dtype=np.int64)
    videos = ["good_a.mp4"] * 4 + ["bad_a.mp4"] * 4
    videos += ["good_b.mp4"] * 4 + ["bad_b.mp4"] * 4

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "train.npz"
        np.savez_compressed(path, features=features, labels=labels)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"shots": [{"video": video} for video in videos]}),
            encoding="utf-8",
        )
        config["data"]["source"] = "npy"
        config["data"]["train_path"] = str(path)
        config["data"]["val_path"] = ""
        config["data"]["validation_split"] = 0.5
        train_loader, val_loader, train_ds, val_ds = make_dataloaders(config)

    assert set(train_ds.groups or ()).isdisjoint(set(val_ds.groups or ()))
    assert set(train_ds.labels.tolist()) == {0, 1}
    assert set(val_ds.labels.tolist()) == {0, 1}
    assert getattr(train_loader, "normalization_stats") is not None
    assert getattr(val_loader, "normalization_stats") is not None


if __name__ == "__main__":
    test_config_loads_and_validates()
    test_synthetic_dataset_shapes()
    test_forward_and_backward()
    test_optimizer_steps()
    test_checkpoint_roundtrip()
    test_device_resolution()
    test_inference_softmax()
    test_feature_normalization_uses_training_statistics()
    test_grouped_split_has_no_video_leakage()
    print("All ml/tests/test_ml_core.py checks passed.")
