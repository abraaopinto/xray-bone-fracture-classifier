from __future__ import annotations

import numpy as np
import torch

from xray_bone_fracture_classifier.interpretability.gradcam import GradCAM
from xray_bone_fracture_classifier.models.transfer import build_transfer_model


def test_gradcam_smoke_cpu() -> None:
    labels = ["no_fracture", "fracture"]
    model = build_transfer_model(arch="resnet18", num_classes=len(labels), pretrained=False)
    model.eval()  # stays on CPU

    # GradCAM expects a tensor [1, C, H, W]
    x = torch.randn(1, 3, 224, 224)

    cam = GradCAM(model=model)  # no 'arch' arg in your implementation
    result = cam(x, target_class=1)
    cam.close()

    assert isinstance(result.cam, np.ndarray)
    assert result.cam.ndim == 2
    assert result.cam.shape == (224, 224)
    assert 0.0 <= float(result.cam.min()) <= 1.0
    assert 0.0 <= float(result.cam.max()) <= 1.0
