"""Regressors, implemented directly because no package index is reachable here.

Each is deliberately LOW CAPACITY. With 18 development samples the binding
constraint is not which family fits best, it is which family can be stopped
from memorising eighteen numbers -- so every model below exposes exactly one
knob that trades fit for smoothness, and `train.py` chooses it by nested
cross-validation rather than by looking at a score.

Four families with genuinely different inductive biases, on purpose. If they
disagree wildly the dataset is too small to identify a model, and that is
itself the finding; if they agree, the answer is not an artefact of one
family's assumptions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class Standardiser:
    """Z-score using TRAINING statistics only.

    Fitting the scaler on all the data before splitting is the quietest form
    of leakage there is -- the test set's mean is a fact about the test set.
    """

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardiser":
        self.mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        # A constant column carries no information; dividing by its zero spread
        # would turn it into infinity and dominate everything else.
        self.sd = np.where(sd < 1e-8, 1.0, sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class ConstantModel:
    """Predicts the training mean. The floor every other model must clear.

    Not a strawman here: elevation on this corpus is compressed around the 0.12
    boundary, so a constant already classifies most shots correctly. Reporting
    accuracy without this number would be reporting the corpus, not the model.
    """

    name = "constant"

    def __init__(self, **_) -> None:
        self.mu = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConstantModel":
        self.mu = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.mu)


class Ridge:
    """Closed-form L2 regression. The knob is alpha.

    With 68 summary features and 18 samples the system is wildly
    underdetermined, so alpha is not a refinement -- it is the only thing
    making the problem solvable at all.
    """

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        # Centre both sides so the intercept is not penalised: shrinking it
        # would pull every prediction toward zero rather than toward the mean.
        self.b = float(np.mean(y))
        Xc, yc = X - np.mean(X, axis=0), y - self.b
        n_f = Xc.shape[1]
        A = Xc.T @ Xc + self.alpha * np.eye(n_f)
        self.w = np.linalg.solve(A, Xc.T @ yc)
        self._mu = np.mean(X, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) @ self.w + self.b


class KNN:
    """Distance-weighted k-nearest neighbours. The knob is k.

    Included because its bias is the opposite of ridge's: no global structure,
    only "shots that look like this one". On a corpus with a handful of
    players that is also its danger -- the nearest neighbour is often the same
    player -- which is precisely why the folds hold out whole groups.
    """

    name = "knn"

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X, self.y = X, y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        k = max(1, min(self.k, len(self.X)))
        for i, row in enumerate(X):
            d = np.sqrt(((self.X - row) ** 2).sum(axis=1))
            idx = np.argsort(d)[:k]
            w = 1.0 / (d[idx] + 1e-6)
            out[i] = float((self.y[idx] * w).sum() / w.sum())
        return out


class BoostedStumps:
    """Gradient boosting on depth-1 trees. The knob is the number of rounds.

    Stumps, not deeper trees: a depth-3 tree can isolate a single one of
    eighteen samples in one split, and would. A stump can only ever say "this
    feature, above or below this value", so capacity grows one small step per
    round and early stopping is meaningful.
    """

    name = "stumps"

    def __init__(self, rounds: int = 30, lr: float = 0.1) -> None:
        self.rounds = int(rounds)
        self.lr = float(lr)
        self.base = 0.0
        self.trees: list = []

    @staticmethod
    def _best_stump(X: np.ndarray, r: np.ndarray) -> tuple:
        best = (None, None, 0.0, 0.0, np.inf)
        for f in range(X.shape[1]):
            col = X[:, f]
            for t in np.unique(np.quantile(col, [0.25, 0.5, 0.75])):
                left = col <= t
                if left.sum() == 0 or (~left).sum() == 0:
                    continue
                lv, rv = r[left].mean(), r[~left].mean()
                err = ((r[left] - lv) ** 2).sum() + ((r[~left] - rv) ** 2).sum()
                if err < best[4]:
                    best = (f, t, lv, rv, err)
        return best

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostedStumps":
        self.base = float(np.mean(y))
        pred = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.rounds):
            r = y - pred
            f, t, lv, rv, _ = self._best_stump(X, r)
            if f is None:
                break
            self.trees.append((f, t, lv, rv))
            pred = pred + self.lr * np.where(X[:, f] <= t, lv, rv)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full(len(X), self.base)
        for f, t, lv, rv in self.trees:
            out = out + self.lr * np.where(X[:, f] <= t, lv, rv)
        return out


class TemporalNet:
    """A small 1-D CNN over the resampled sequence. The knob is epochs.

    THE ONLY MODEL HERE THAT READS THE SHOT AS A SEQUENCE, which is the whole
    reason the project wanted a temporal model: the others see summaries and
    cannot tell a smooth rise from a jerky one that reaches the same peak.

    Sized against the data rather than against the task: two convolutions, 8
    channels, global average pooling. Even so it has more parameters than the
    dataset has numbers, so weight decay, dropout and early stopping are doing
    most of the work, and it is entirely possible this loses to ridge. That
    result would be worth having.

    Global average pooling rather than a flatten: flattening would give the
    model a separate weight for every timestep, and with 18 samples it would
    learn "the answer for the shot with this exact shape at t=17".
    """

    name = "tempnet"

    def __init__(self, epochs: int = 200, hidden: int = 8, seed: int = 0) -> None:
        self.epochs = int(epochs)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.net = None

    def _build(self, n_ch: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv1d(n_ch, self.hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TemporalNet":
        import torch

        torch.manual_seed(self.seed)
        # (N, T, C) -> (N, C, T): Conv1d convolves over the last axis, and that
        # axis must be TIME.
        xb = torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
        yb = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        self.net = self._build(X.shape[2])
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-2, weight_decay=1e-2)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.05)

        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xb), yb)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        xb = torch.tensor(np.transpose(X, (0, 2, 1)), dtype=torch.float32)
        with torch.no_grad():
            return self.net(xb).view(-1).numpy()
