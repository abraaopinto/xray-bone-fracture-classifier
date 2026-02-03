from __future__ import annotations

import torch


def resolve_device(device_str: str | None) -> torch.device:
    """Resolve execution device consistently across train and inference."""
    device_str = (device_str or "auto").strip().lower()

    if device_str in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str in {"directml", "dml"}:
        try:
            import torch_directml  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Device 'directml' requer dependência opcional. Instale com: pip install -e '.[directml]'"
            ) from e
        return torch_directml.device()

    raise ValueError(f"Device inválido: {device_str}")
