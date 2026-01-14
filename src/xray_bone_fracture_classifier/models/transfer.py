"""Transfer learning models using torchvision."""

from __future__ import annotations

from torch import nn
from torchvision import models


def build_transfer_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a transfer-learning model with a custom classification head."""
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    raise ValueError("Unsupported arch. Use: baseline|resnet18|resnet50|efficientnet_b0")
