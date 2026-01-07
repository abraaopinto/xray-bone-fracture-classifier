"""Configuration utilities for training/evaluation/inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import time


@dataclass(frozen=True)
class TrainConfig:
    """Training configuration."""
    data_dir: str
    arch: str = "resnet18"          # baseline|resnet18|resnet50|efficientnet_b0
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    device: Optional[str] = None    # cpu|cuda|None(auto)
    early_stopping_patience: int = 5
    label_smoothing: float = 0.0


def make_run_dir(root: Path, prefix: str = "run") -> Path:
    """Create a timestamped run directory under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a JSON file with UTF-8 encoding."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_config(run_dir: Path, cfg: TrainConfig) -> Path:
    """Persist training config as config.json inside run_dir."""
    p = run_dir / "config.json"
    save_json(p, asdict(cfg))
    return p