"""Smoke check for ml.utils.device — run: python ml/tests/smoke_device.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.utils.config import load_train_config
from ml.utils.device import (
    collect_device_info_from_config,
    format_device_report,
    resolve_device,
)


def main() -> int:
    config = load_train_config()
    info = collect_device_info_from_config(config)
    print(format_device_report(info))
    print("---")

    auto = resolve_device("auto")
    cpu = resolve_device("cpu")
    print(f"resolve_device('auto') -> {auto}")
    print(f"resolve_device('cpu')  -> {cpu}")
    assert str(cpu) == "cpu"
    print("device smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
