from __future__ import annotations

import torch

from xray_bone_fracture_classifier.models.transfer import build_transfer_model


def test_forward_pass_resnet18_cpu(device_cpu: torch.device) -> None:
    num_classes = 2
    model = build_transfer_model(arch="resnet18", num_classes=num_classes, pretrained=False)
    model = model.to(device_cpu)
    model.eval()

    x = torch.randn(4, 3, 224, 224, device=device_cpu)
    with torch.no_grad():
        y = model(x)

    assert y.shape == (4, num_classes)
