"""Smoke check for ml.utils.config — run: python ml/tests/smoke_config.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.utils.config import load_train_config, summarize_config, validate_train_config


def main() -> int:
    config = load_train_config()
    print(summarize_config(config))
    print("---")
    assert config["model"]["input_size"] == config["data"]["feature_dim"]
    assert config["model"]["output_size"] == config["data"]["num_classes"]

    bad = load_train_config()
    bad["model"]["input_size"] = 999
    try:
        validate_train_config(bad)
    except ValueError as exc:
        print(f"validation correctly rejected mismatch: {exc}")
    else:
        print("ERROR: expected ValueError for input_size mismatch")
        return 1

    print("config smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
