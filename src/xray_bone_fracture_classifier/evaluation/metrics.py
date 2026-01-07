"""Metrics for classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

@dataclass(frozen=True)
class EvalOutputs:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalOutputs:
    model.eval()
    ys, ps, probs = [], [], []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = softmax(logits).detach().cpu().numpy()
            pred = np.argmax(prob, axis=1)
            ys.append(yb.numpy())
            ps.append(pred)
            probs.append(prob)

    return EvalOutputs(
        y_true=np.concatenate(ys, axis=0),
        y_pred=np.concatenate(ps, axis=0),
        y_prob=np.concatenate(probs, axis=0),
    )

def compute_metrics(y_true, y_pred, num_classes: int) -> dict:
    labels_all = list(range(num_classes))

    rep_present = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_all = classification_report(y_true, y_pred, labels=labels_all, output_dict=True, zero_division=0)

    acc = float((y_true == y_pred).mean())

    return {
        "accuracy": acc,
        "f1_macro_present": float(rep_present["macro avg"]["f1-score"]),
        "f1_macro_all": float(rep_all["macro avg"]["f1-score"]),
        "num_classes_all": num_classes,
        "num_classes_present_in_test": int(len(set(y_true.tolist()))),
    }

def compute_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    labels: List[int] | None = None,
) -> Tuple[Dict, np.ndarray]:
    """
    Compute sklearn classification report and confusion matrix.

    Key point:
    - If some classes are missing in y_true/y_pred (common in small test splits),
      sklearn needs an explicit `labels` list to align with `target_names`.
    """
    if labels is None:
        labels = list(range(len(target_names)))

    rep = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[target_names[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return rep, cm