"""Training / validation step helpers.

Keeps ``train.py`` thin: one place for forward, loss, backward, metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def _accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of correct argmax predictions in a batch."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return float(correct) / max(labels.numel(), 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: GradScaler | None,
    gradient_clip: float,
    mixed_precision: bool,
    log_every_n_steps: int = 10,
) -> Dict[str, float]:
    """One full pass over the training set.

    Gradient flow (concept)
    -----------------------
    1. Forward: logits = model(x)
    2. Loss: scalar measuring wrongness vs labels
    3. backward: autograd fills ``.grad`` on each parameter
    4. clip: shrink huge gradients
    5. optimizer.step: AdamW updates weights using grads + moments
    6. zero_grad: clear grads before the next batch
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0

    use_amp = bool(mixed_precision) and device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for step, (features, labels) in enumerate(loader, start=1):
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        batch_acc = _accuracy_from_logits(logits.detach(), labels)
        total_loss += float(loss.detach().item())
        total_acc += batch_acc
        steps += 1

        if log_every_n_steps > 0 and step % log_every_n_steps == 0:
            print(
                f"  step {step:04d}/{len(loader):04d} "
                f"loss={loss.item():.4f} acc={batch_acc:.3f}"
            )

    return {
        "loss": total_loss / max(steps, 1),
        "accuracy": total_acc / max(steps, 1),
    }


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    mixed_precision: bool,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """Validation pass — no dropout randomness, no weight updates.

    Returns
    -------
    metrics:
        mean loss / accuracy
    all_logits:
        concatenated logits on CPU ``(N, C)``
    all_labels:
        concatenated labels on CPU ``(N,)``
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    steps = 0
    logits_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    use_amp = bool(mixed_precision) and device.type == "cuda"
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=amp_device, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        total_loss += float(loss.item())
        total_acc += _accuracy_from_logits(logits, labels)
        steps += 1
        logits_chunks.append(logits.detach().cpu())
        label_chunks.append(labels.detach().cpu())

    metrics = {
        "loss": total_loss / max(steps, 1),
        "accuracy": total_acc / max(steps, 1),
    }
    return metrics, torch.cat(logits_chunks, dim=0), torch.cat(label_chunks, dim=0)
