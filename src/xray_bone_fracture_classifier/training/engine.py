"""Training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> EpochResult:
    """Run one epoch. If optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += (torch.argmax(logits, dim=1) == yb).sum().item()
        total_seen += bs

    return EpochResult(loss=float(total_loss / max(1, total_seen)),
                      accuracy=float(total_correct / max(1, total_seen)))


def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 5,
) -> Tuple[nn.Module, Dict[str, list]]:
    """Train model with early stopping on validation loss."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_loss = float("inf")
    patience = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(1, epochs + 1):
        tr = run_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va = run_one_epoch(model, loaders["valid"], criterion, None, device)

        history["train_loss"].append(tr.loss)
        history["train_acc"].append(tr.accuracy)
        history["val_loss"].append(va.loss)
        history["val_acc"].append(va.accuracy)

        if va.loss < best_val_loss - 1e-6:
            best_val_loss = va.loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
