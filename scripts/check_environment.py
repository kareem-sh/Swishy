"""Check Swichy's Python, CUDA, and YOLO environment.

Usage:
    python scripts/check_environment.py
    python scripts/check_environment.py --require-cuda
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Swichy runtime dependencies")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit with an error when CUDA is unavailable",
    )
    args = parser.parse_args()

    print(f"Python:       {platform.python_version()}")
    print(f"Executable:   {sys.executable}")
    print(f"Ultralytics:  {package_version('ultralytics')}")
    print(f"MediaPipe:    {package_version('mediapipe')}")
    print(f"OpenCV:       {package_version('opencv-python')}")

    try:
        import torch
    except ImportError:
        print("PyTorch:      not installed")
        return 1

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'none (CPU build)'}")
    print(f"CUDA ready:   {cuda_available}")
    if cuda_available:
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory:   {memory_gb:.1f} GB")
    else:
        print("GPU:          CPU fallback")

    if args.require_cuda and not cuda_available:
        print(
            "\nERROR: CUDA was required but is unavailable. "
            "Follow docs/GPU_YOLO_SETUP.md."
        )
        return 2

    try:
        from ball.yolo_model import resolve_device

        print(f"Swichy device:{resolve_device('auto')!s:>8}")
    except Exception as exc:
        print(f"Swichy device check failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
