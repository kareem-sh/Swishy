"""Shot-quality MLP (version 1 — educational, no attention/CNN/LSTM).

Architecture (from train.yaml)
------------------------------
Input (F,)
  → Linear(F → H)
  → BatchNorm1d(H)
  → ReLU
  → Dropout(p)
  → Linear(H → H)
  → BatchNorm1d(H)
  → ReLU
  → Dropout(p)
  → Linear(H → C)   # logits, NOT probabilities

Why each layer
--------------
Linear     : learn a weight matrix; core of the MLP.
BatchNorm  : stabilize activations; allows higher learning rates.
ReLU       : nonlinearity so the net is not just one big linear map.
Dropout    : randomly zero units in train mode → less overfitting.
Final Linear : C raw scores (logits). Softmax is applied only for probs.

TODO:
    - Increase hidden layer size after real data arrives.
    - Try deeper architecture / residual connections.
    - Export ONNX model.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class ShotQualityMLP(nn.Module):
    """Multi-layer perceptron for shot-quality classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.3,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        in_features = input_size
        for _ in range(num_hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden_size

        # Classifier head — no softmax here (CrossEntropyLoss includes it).
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape ``(batch_size, input_size)``.

        Returns
        -------
        logits:
            Shape ``(batch_size, output_size)``.
            Each row is unnormalized class scores.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, F), got {tuple(x.shape)}")
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected feature dim {self.input_size}, got {x.shape[1]}"
            )
        return self.net(x)


def build_model_from_config(config: Mapping[str, Any]) -> ShotQualityMLP:
    """Construct ``ShotQualityMLP`` using the ``model`` section of train.yaml."""
    model_cfg = config["model"]
    return ShotQualityMLP(
        input_size=int(model_cfg["input_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        output_size=int(model_cfg["output_size"]),
        dropout=float(model_cfg["dropout"]),
        num_hidden_layers=int(model_cfg["num_hidden_layers"]),
    )
