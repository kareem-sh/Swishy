"""Device selection and environment reporting for Swichy ML.

WHY THIS FILE EXISTS
--------------------
Training code should never hardcode ``device = "cuda"``.
Machines differ:

  - Karim's current box: NVIDIA GPU (previously verified GTX 1650)
  - Salah's training box: may be RTX 3070 / another CUDA GPU / CPU-only
  - Apple Silicon: may expose MPS instead of CUDA

This module answers one question:

    Which ``torch.device`` should tensors and the model live on?

It also prints a clear environment report so you can confirm CUDA is
actually being used before waiting hours for a bad CPU-only run.

HOW CUDA IS USED (concept)
--------------------------
1. Model parameters start on CPU when you construct ``nn.Module``.
2. ``model.to(device)`` copies weights to GPU memory (VRAM).
3. Each batch ``x = x.to(device)`` copies features to VRAM.
4. Matmul / Linear / BatchNorm run as CUDA kernels on the GPU.
5. Gradients stay on GPU; ``optimizer.step()`` updates GPU weights.
6. Only metrics / checkpoints need to come back to CPU when saving.

TODO:
    - Log device info into TensorBoard text / checkpoint metadata.
    - Add a ``--force-cpu`` CLI flag for debugging on Salah's machine.
    - Detect multi-GPU and document DataParallel / DDP as a later upgrade.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Any, Mapping

# ``torch`` is imported lazily inside functions that need it so that
# ``import ml.utils.device`` does not fail in environments where torch
# is temporarily broken — except when you actually ask for a device.


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable snapshot of the compute environment.

    Attributes
    ----------
    device:
        The ``torch.device`` to use for model + batches
        (e.g. ``device(type='cuda', index=0)``).
    device_type:
        Short label: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    python_version:
        e.g. ``"3.13.5"``.
    torch_version:
        e.g. ``"2.11.0+cu128"``.
    torchvision_version:
        Installed torchvision version, or ``"not installed"``.
    cuda_version:
        CUDA runtime version bundled with the PyTorch wheel, or ``None``.
    gpu_name:
        Human-readable GPU name, or ``None`` on CPU/MPS.
    gpu_memory_gb:
        Total VRAM in GiB, or ``None`` when not a CUDA device.
    """

    device: Any  # torch.device — typed as Any to avoid hard import at type-check time
    device_type: str
    python_version: str
    torch_version: str
    torchvision_version: str
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_gb: float | None


def _torchvision_version() -> str:
    """Return torchvision version string without crashing if missing."""
    try:
        import torchvision

        return str(torchvision.__version__)
    except ImportError:
        return "not installed"


def resolve_device(preferred: str = "auto", cuda_device_index: int = 0) -> Any:
    """Pick a ``torch.device`` from a preference string.

    Parameters
    ----------
    preferred:
        ``"auto"`` | ``"cuda"`` | ``"mps"`` | ``"cpu"``.
        - auto: CUDA if available, else MPS, else CPU
        - cuda: GPU if available, otherwise fall back to CPU with a warning path
          handled by the caller via ``DeviceInfo``
        - mps: Apple Metal if available, else CPU
        - cpu: always CPU
    cuda_device_index:
        Which GPU to use when multiple CUDA devices exist (0 = first GPU).

    Returns
    -------
    torch.device

    Common beginner mistake
    -----------------------
    Writing ``model.cuda()`` unconditionally. On a machine without CUDA this
    raises. Prefer ``model.to(resolve_device(...))``.
    """
    import torch

    choice = (preferred or "auto").strip().lower()

    if choice in {"auto", ""}:
        if torch.cuda.is_available():
            return torch.device("cuda", cuda_device_index)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", cuda_device_index)
        # Explicit request but no GPU — fall back so training still runs.
        return torch.device("cpu")

    if choice == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device preference '{preferred}'. "
        f"Use auto | cuda | mps | cpu."
    )


def collect_device_info(
    preferred: str = "auto",
    cuda_device_index: int = 0,
) -> DeviceInfo:
    """Resolve the device and gather printable environment facts.

    This does **not** allocate a large tensor; it only queries drivers /
    PyTorch. Safe to call at the start of every training run.
    """
    import torch

    device = resolve_device(preferred=preferred, cuda_device_index=cuda_device_index)
    device_type = device.type

    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None

    if device_type == "cuda" and torch.cuda.is_available():
        cuda_version = torch.version.cuda
        index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
        # total_memory is bytes → convert to GiB
        total_bytes = torch.cuda.get_device_properties(index).total_memory
        gpu_memory_gb = total_bytes / (1024 ** 3)

    return DeviceInfo(
        device=device,
        device_type=device_type,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torchvision_version=_torchvision_version(),
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


def collect_device_info_from_config(config: Mapping[str, Any]) -> DeviceInfo:
    """Convenience wrapper: read ``device`` section from ``train.yaml`` config."""
    section = config.get("device", {})
    preferred = str(section.get("preferred", "auto"))
    index = int(section.get("cuda_device_index", 0))
    return collect_device_info(preferred=preferred, cuda_device_index=index)


def format_device_report(info: DeviceInfo) -> str:
    """Build the pre-training environment block for the console.

    Example output
    --------------
    ::

        Python:       3.13.5
        Torch:        2.11.0+cu128
        Torchvision:  0.22.0
        CUDA:         12.8
        Device:       cuda:0
        GPU:          NVIDIA GeForce GTX 1650
        GPU memory:   4.0 GB
    """
    memory_line = (
        f"{info.gpu_memory_gb:.1f} GB"
        if info.gpu_memory_gb is not None
        else "n/a"
    )
    lines = [
        f"Python:       {info.python_version}",
        f"Executable:   {sys.executable}",
        f"Torch:        {info.torch_version}",
        f"Torchvision:  {info.torchvision_version}",
        f"CUDA:         {info.cuda_version or 'none (CPU/MPS build or unavailable)'}",
        f"Device:       {info.device}",
        f"GPU:          {info.gpu_name or 'n/a'}",
        f"GPU memory:   {memory_line}",
    ]
    return "\n".join(lines)


def print_device_report(info: DeviceInfo | None = None) -> DeviceInfo:
    """Print the environment report and return ``DeviceInfo``.

    If ``info`` is omitted, uses ``preferred="auto"``.
    """
    if info is None:
        info = collect_device_info(preferred="auto")
    print(format_device_report(info))
    return info


def count_trainable_parameters(module: Any) -> int:
    """Count parameters with ``requires_grad=True``.

    Why this matters
    ----------------
    Trainable parameter count ≈ model capacity.
    For our smoke-test MLP (32 → 128 → 128 → 5) you should see on the order
    of tens of thousands of parameters — tiny vs YOLO/transformers.

    ``numel()`` = number of scalar elements in a tensor
    (e.g. a weight matrix of shape ``[128, 32]`` has 4096 parameters).
    """
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def format_parameter_report(module: Any) -> str:
    """Human-readable parameter summary for training logs."""
    trainable = count_trainable_parameters(module)
    total = sum(param.numel() for param in module.parameters())
    return (
        f"Trainable parameters: {trainable:,}\n"
        f"Total parameters:     {total:,}"
    )


# TODO: Add empty_cuda_cache() helper for long hyperparameter sweeps.
# TODO: Add a context manager that times a CUDA event pair for profiling.
