"""Classification metrics explained for evaluation + early stopping.

Metrics
-------
Accuracy:
    correct / total. Easy to read; misleading if classes are imbalanced.

Precision (per class):
    TP / (TP + FP). \"When we predict class k, how often are we right?\"

Recall (per class):
    TP / (TP + FN). \"Of all true class-k shots, how many did we find?\"

F1:
    Harmonic mean of precision and recall.

Macro average:
    Unweighted mean across classes — treats rare classes equally.

TODO:
    - Add calibration plots (confidence vs accuracy).
    - Log per-class metrics to TensorBoard as scalars.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def confusion_matrix(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Return matrix ``C`` where ``C[i, j]`` = true i predicted j."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            matrix[t, p] += 1
    return matrix


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def classification_scores(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    num_classes: int,
) -> Dict[str, float | np.ndarray | str]:
    """Compute accuracy, macro precision/recall/F1, and a text report."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = confusion_matrix(y_true, y_pred, num_classes)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)

    precision = _safe_div(tp, predicted)
    recall = _safe_div(tp, support)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    # Macro: mean over classes that appear OR all classes (use all for stability).
    macro_precision = float(precision.mean())
    macro_recall = float(recall.mean())
    macro_f1 = float(f1.mean())

    lines = [
        f"{'class':>8} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}",
    ]
    for idx in range(num_classes):
        lines.append(
            f"{idx:8d} {precision[idx]:10.3f} {recall[idx]:10.3f} "
            f"{f1[idx]:10.3f} {int(support[idx]):10d}"
        )
    lines.append("")
    lines.append(f"accuracy: {accuracy:.4f}")
    lines.append(
        f"macro avg precision={macro_precision:.4f} "
        f"recall={macro_recall:.4f} f1={macro_f1:.4f}"
    )
    report = "\n".join(lines)

    return {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "confusion_matrix": cm,
        "report": report,
    }
