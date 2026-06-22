import yaml
from pathlib import Path
from typing import Any, Dict

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_yaml(filename: str) -> Dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
