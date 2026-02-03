from __future__ import annotations

import numpy as np

from xray_bone_fracture_classifier.evaluation.metrics import compute_metrics


def test_compute_metrics_contains_required_keys() -> None:
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    out = compute_metrics(y_true, y_pred, num_classes=2)

    required = {
        "accuracy",
        "precision_macro_present",
        "recall_macro_present",
        "f1_macro_present",
        "precision_macro_all",
        "recall_macro_all",
        "f1_macro_all",
        "num_classes_all",
        "num_classes_present_in_test",
    }

    missing = required.difference(out.keys())
    assert not missing, f"Missing keys: {missing}"

    assert 0.0 <= out["accuracy"] <= 1.0
