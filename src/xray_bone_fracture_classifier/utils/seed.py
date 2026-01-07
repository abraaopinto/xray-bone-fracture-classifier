"""Reproducibility helpers."""

from __future__ import annotations

import os
import random
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
