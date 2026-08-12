from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@lru_cache(maxsize=None)
def _load_yaml_cached(filename: str) -> Dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_yaml(filename: str) -> Dict[str, Any]:
    """Load config once, returning an independent copy to each caller."""
    return deepcopy(_load_yaml_cached(filename))


def clear_config_cache() -> None:
    """Clear cached YAML, mainly for tests or runtime config editors."""
    _load_yaml_cached.cache_clear()
