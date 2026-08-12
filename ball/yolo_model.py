"""Custom basketball YOLO loader (ball + hoop + player + referee).

Uses the E-BARD fine-tuned weights (ball + hoop). Ultralytics package
should be latest (YOLO26-capable). Device defaults to CUDA when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

_BALL_ALIASES = {"ball", "basketball", "sports ball", "sports_ball"}
_RIM_ALIASES = {"rim", "hoop", "basketball hoop", "basketball_hoop", "basket"}

DEFAULT_HF_REPO = "GabrieleGiudici/E-BARD-detection-models"
DEFAULT_HF_FILE = "BODD_yolov8n_0001.pt"
DEFAULT_LOCAL = Path("models/basketball_yolov8n.pt")

# Preferred base for *future* fine-tuning (latest Ultralytics generation)
TRAIN_BASE_MODEL = "yolo26n.pt"

_ROOT = Path(__file__).resolve().parent.parent


def resolve_device(preferred: str | int | None = "auto") -> str | int:
    """Pick CUDA GPU when available; otherwise CPU.

    preferred:
      auto | cuda | cpu | 0 | 1 | ...
    """
    if preferred is None or str(preferred).lower() in ("auto", ""):
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    text = str(preferred).lower()
    if text == "cuda":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if text == "cpu":
        return "cpu"
    if text.isdigit():
        return int(text)
    return preferred


def resolve_model_path(
    model_path: str | Path,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_file: str = DEFAULT_HF_FILE,
) -> Path:
    """Return a local .pt path, downloading the basketball model if needed."""
    path = Path(model_path)
    if not path.is_absolute():
        path = _ROOT / path

    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=hf_repo,
                filename=hf_file,
                local_dir=str(path.parent),
            )
        )
        if downloaded.resolve() != path.resolve() and not path.exists():
            path.write_bytes(downloaded.read_bytes())
        if path.is_file():
            return path
        if downloaded.is_file():
            return downloaded
    except Exception as exc:
        print(f"Warning: could not download basketball YOLO ({hf_repo}/{hf_file}): {exc}")

    raise FileNotFoundError(
        f"Basketball YOLO weights not found at {path}. "
        f"Expected auto-download from {hf_repo}/{hf_file}."
    )


def load_basketball_yolo(
    model_path: str | Path = DEFAULT_LOCAL,
    hf_repo: str = DEFAULT_HF_REPO,
    hf_file: str = DEFAULT_HF_FILE,
) -> Tuple[Any, List[int], List[int], Dict[int, str]]:
    """Load custom ball+rim YOLO and return (model, ball_ids, rim_ids, names)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required. Install with: pip install -U ultralytics"
        ) from exc

    path = resolve_model_path(model_path, hf_repo=hf_repo, hf_file=hf_file)
    model = YOLO(str(path), task="detect")
    names = {int(k): str(v).lower() for k, v in (model.names or {}).items()}
    ball_ids = [i for i, n in names.items() if n in _BALL_ALIASES]
    rim_ids = [i for i, n in names.items() if n in _RIM_ALIASES]
    if not ball_ids:
        ball_ids = [i for i, n in names.items() if "ball" in n]

    print(
        f"Basketball YOLO: {path.name} | ball={ball_ids} rim={rim_ids} | "
        f"classes={names}"
    )
    return model, ball_ids, rim_ids, names
