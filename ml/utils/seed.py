"""Reproducibility helpers for Swichy ML.

WHY THIS FILE EXISTS
--------------------
Neural-net training uses randomness in:

  - weight initialization
  - DataLoader shuffling
  - Dropout masks
  - CUDA algorithms (sometimes)

``set_seed`` forces those sources to share one integer seed so Salah can
rerun an experiment and get comparable results.

TODO:
    - Store the seed inside every checkpoint under key \"seed\".
    - Log seed + git commit together in TensorBoard.
"""

from __future__ import annotations

import os
import random
from typing import Mapping


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed:
        Non-negative integer from ``train.yaml`` → ``reproducibility.random_seed``.
    deterministic:
        If True, ask cuDNN for deterministic algorithms (slightly slower,
        more reproducible). Some ops may still be nondeterministic on GPU.

    Notes
    -----
    Call this **once** at the start of ``train.py``, before creating the model
    or DataLoaders.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster when input sizes are fixed; less reproducible.
        torch.backends.cudnn.benchmark = True


def set_seed_from_config(config: Mapping) -> int:
    """Read seed settings from a loaded train config and apply them.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """
    section = config.get("reproducibility", {})
    seed = int(section.get("random_seed", 42))
    deterministic = bool(section.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)
    return seed
