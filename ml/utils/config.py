"""ML training configuration loader.

WHY THIS FILE EXISTS
--------------------
``ml/configs/train.yaml`` stores every hyperparameter as text.
This module is the *only* place that should open that YAML for training,
evaluation, and inference scripts.

That means:
  - no duplicated ``yaml.safe_load`` calls scattered across the repo
  - one validation gate that catches size mismatches early
  - paths resolve relative to the Swichy repo root, not the caller's CWD

HOW PYTHON EXECUTES THIS MODULE
-------------------------------
1. ``import ml.utils.config`` loads this file once (cached in ``sys.modules``).
2. Calling ``load_train_config()`` reads disk, parses YAML to a ``dict``,
   validates required keys, then returns the dict.
3. Later files (``train.py``, ``evaluate.py``) do::

       from ml.utils.config import load_train_config
       config = load_train_config()

TODO:
    - Add a typed dataclass / pydantic model if configs grow complex.
    - Support merging CLI overrides on top of YAML
      (e.g. ``--learning-rate 3e-4``).
    - Load experiment-specific configs like ``train_basketball.yaml``.
    - Record the absolute config path inside every checkpoint.
"""

from __future__ import annotations

# ``annotations`` makes type hints strings at runtime (PEP 563/649 style).
# Benefit: we can write ``dict[str, Any]`` cleanly on older patterns and
# forward-reference types without quotes in many cases.

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# __file__ is this module's path on disk, e.g.
#   C:/Users/.../Swichy/ml/utils/config.py
# parents[0] = ml/utils
# parents[1] = ml
# parents[2] = Swichy repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "ml" / "configs" / "train.yaml"

# Sections that every training config must contain.
# If someone deletes ``optimizer`` from the YAML, we fail immediately with a
# clear error instead of crashing later inside AdamW construction.
_REQUIRED_SECTIONS: Sequence[str] = (
    "experiment",
    "reproducibility",
    "device",
    "data",
    "model",
    "training",
    "optimizer",
    "scheduler",
    "checkpoint",
    "logging",
    "evaluation",
    "inference",
)

# Keys that must exist inside specific sections (minimum contract).
_REQUIRED_KEYS: Mapping[str, Sequence[str]] = {
    "reproducibility": ("random_seed", "deterministic"),
    "device": ("preferred",),
    "data": (
        "source",
        "feature_dim",
        "num_classes",
        "validation_split",
        "num_workers",
    ),
    "model": (
        "input_size",
        "hidden_size",
        "num_hidden_layers",
        "output_size",
        "dropout",
    ),
    "training": (
        "epochs",
        "batch_size",
        "gradient_clip",
        "mixed_precision",
        "early_stopping_patience",
        "early_stopping_metric",
    ),
    "optimizer": ("name", "learning_rate", "weight_decay"),
    "scheduler": ("name",),
    "checkpoint": ("dir", "best_filename", "last_filename", "resume"),
}


def get_repo_root() -> Path:
    """Return the absolute path to the Swichy repository root.

    Example
    -------
    >>> root = get_repo_root()
    >>> (root / "ml" / "configs" / "train.yaml").is_file()
    True
    """
    return _REPO_ROOT


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a config path to an absolute filesystem path.

    Parameters
    ----------
    config_path:
        - ``None`` → default ``ml/configs/train.yaml``
        - absolute path → used as-is
        - relative path → joined onto the repo root (not the process CWD)

    Why repo-root relative?
        Running ``python ml/training/train.py`` from different folders would
        otherwise break ``open("ml/configs/train.yaml")``. Anchoring to the
        repo root makes scripts location-independent.
    """
    if config_path is None:
        return _DEFAULT_CONFIG_PATH

    path = Path(config_path)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _read_yaml(path: Path) -> MutableMapping[str, Any]:
    """Open a YAML file and return a mutable mapping.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is empty or does not parse to a mapping (dict).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"ML config not found: {path}. "
            f"Expected train.yaml under ml/configs/."
        )

    # encoding="utf-8" avoids Windows locale surprises (cp1252).
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    # ``safe_load`` returns None for an empty file.
    if raw is None:
        raise ValueError(f"ML config is empty: {path}")

    if not isinstance(raw, MutableMapping):
        raise ValueError(
            f"ML config must be a YAML mapping (dict) at the top level, "
            f"got {type(raw).__name__}: {path}"
        )

    return raw


def _require_section(
    config: Mapping[str, Any],
    section: str,
    path: Path,
) -> Mapping[str, Any]:
    """Return ``config[section]`` or raise a clear KeyError-style message."""
    if section not in config:
        raise ValueError(
            f"Missing required section '{section}' in {path}. "
            f"Required sections: {list(_REQUIRED_SECTIONS)}"
        )
    value = config[section]
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Section '{section}' in {path} must be a mapping, "
            f"got {type(value).__name__}."
        )
    return value


def validate_train_config(
    config: Mapping[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Validate structure and cross-field consistency.

    Checks performed
    ----------------
    1. All required top-level sections exist.
    2. Critical keys inside those sections exist.
    3. ``model.input_size == data.feature_dim``
       (a Linear layer's ``in_features`` must match the feature vector length).
    4. ``model.output_size == data.num_classes``
       (CrossEntropyLoss expects ``logits.shape[-1] == num_classes``).
    5. Basic numeric ranges (positive sizes, dropout in [0, 1), etc.).

    Returns
    -------
    dict
        A *deep copy* of the config so callers cannot accidentally mutate the
        cached/shared structure if we add caching later.
    """
    label = path if path is not None else Path("<in-memory-config>")

    for section in _REQUIRED_SECTIONS:
        _require_section(config, section, label)

    for section, keys in _REQUIRED_KEYS.items():
        block = config[section]
        missing = [key for key in keys if key not in block]
        if missing:
            raise ValueError(
                f"Section '{section}' in {label} is missing keys: {missing}"
            )

    data = config["data"]
    model = config["model"]
    training = config["training"]
    optimizer = config["optimizer"]

    feature_dim = int(data["feature_dim"])
    num_classes = int(data["num_classes"])
    input_size = int(model["input_size"])
    output_size = int(model["output_size"])

    if input_size != feature_dim:
        raise ValueError(
            f"model.input_size ({input_size}) must equal "
            f"data.feature_dim ({feature_dim}) in {label}. "
            f"The first Linear layer expects in_features == feature vector length."
        )

    if output_size != num_classes:
        raise ValueError(
            f"model.output_size ({output_size}) must equal "
            f"data.num_classes ({num_classes}) in {label}. "
            f"CrossEntropyLoss compares class index 0..C-1 to C logits."
        )

    if feature_dim < 1 or num_classes < 2:
        raise ValueError(
            f"Need feature_dim >= 1 and num_classes >= 2, got "
            f"feature_dim={feature_dim}, num_classes={num_classes}."
        )

    dropout = float(model["dropout"])
    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            f"model.dropout must be in [0.0, 1.0), got {dropout}."
        )

    if int(training["epochs"]) < 1:
        raise ValueError("training.epochs must be >= 1.")

    if int(training["batch_size"]) < 1:
        raise ValueError("training.batch_size must be >= 1.")

    if float(training["gradient_clip"]) <= 0.0:
        raise ValueError("training.gradient_clip must be > 0.")

    if float(optimizer["learning_rate"]) <= 0.0:
        raise ValueError("optimizer.learning_rate must be > 0.")

    split = float(data["validation_split"])
    if not 0.0 <= split < 1.0:
        raise ValueError(
            f"data.validation_split must be in [0.0, 1.0), got {split}."
        )

    metric = str(training["early_stopping_metric"]).lower()
    if metric not in {"accuracy", "f1_macro", "loss"}:
        raise ValueError(
            f"training.early_stopping_metric must be one of "
            f"accuracy | f1_macro | loss, got '{metric}'."
        )

    preferred = str(config["device"]["preferred"]).lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            f"device.preferred must be auto | cuda | mps | cpu, got '{preferred}'."
        )

    # Deep copy: callers may set config["training"]["epochs"] = 1 in tests
    # without mutating a shared object if we later cache loads.
    return deepcopy(dict(config))


def load_train_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the ML training YAML.

    Parameters
    ----------
    config_path:
        Optional path to a YAML file. ``None`` uses ``ml/configs/train.yaml``.

    Returns
    -------
    dict[str, Any]
        Validated nested configuration dictionary.

    Example
    -------
    >>> config = load_train_config()
    >>> config["optimizer"]["learning_rate"]
    0.001
    >>> config["model"]["hidden_size"]
    128
    """
    path = resolve_config_path(config_path)
    raw = _read_yaml(path)
    return validate_train_config(raw, path=path)


def summarize_config(config: Mapping[str, Any]) -> str:
    """Build a one-screen human summary for training logs.

    Useful at the start of ``train.py`` so TensorBoard runs and console
    output show what hyperparameters were actually used.
    """
    lines = [
        f"experiment:     {config['experiment'].get('name', '<unnamed>')}",
        f"seed:           {config['reproducibility']['random_seed']}",
        f"device:         {config['device']['preferred']}",
        f"data.source:    {config['data']['source']}",
        f"feature_dim:    {config['data']['feature_dim']}",
        f"num_classes:    {config['data']['num_classes']}",
        f"hidden_size:    {config['model']['hidden_size']}",
        f"dropout:        {config['model']['dropout']}",
        f"epochs:         {config['training']['epochs']}",
        f"batch_size:     {config['training']['batch_size']}",
        f"lr:             {config['optimizer']['learning_rate']}",
        f"weight_decay:   {config['optimizer']['weight_decay']}",
        f"scheduler:      {config['scheduler']['name']}",
        f"amp:            {config['training']['mixed_precision']}",
    ]
    return "\n".join(lines)


# TODO: Add load_train_config_with_overrides(cli_args) for hyperparameter sweeps.
# TODO: Serialize the exact config dict into each .pt checkpoint under key "config".
