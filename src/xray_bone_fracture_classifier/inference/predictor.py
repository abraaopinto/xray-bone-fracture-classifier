"""Single-image prediction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json

import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from ..data.dataloaders import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    idx_to_class: Dict[int, str]
    class_to_idx: Dict[str, int]
    transform: transforms.Compose
    img_size: int
    device: torch.device


def load_bundle(model_dir: Path, device: str | None = None) -> ModelBundle:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    idx_to_class = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    idx_to_class = {int(k): str(v) for k, v in idx_to_class.items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}

    arch = str(cfg["arch"])
    img_size = int(cfg.get("img_size", 224))

    from ..models.baseline_cnn import BaselineCNN
    from ..models.transfer import build_transfer_model

    num_classes = len(idx_to_class)

    if arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_transfer_model(arch=arch, num_classes=num_classes, pretrained=False)

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    state = torch.load(model_dir / "model.pt", map_location=dev)
    model.load_state_dict(state)
    model.to(dev).eval()

    transform = _eval_transform(img_size)

    return ModelBundle(
        model=model,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        transform=transform,
        img_size=img_size,
        device=dev,
    )

def _eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def predict_image(bundle: ModelBundle, image_path: Path, topk: int = 3) -> List[Tuple[str, float]]:
    img = Image.open(image_path).convert("RGB")
    x = bundle.transform(img).unsqueeze(0).to(bundle.device)

    with torch.no_grad():
        logits = bundle.model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    order = prob.argsort()[::-1][:topk]
    return [(bundle.idx_to_class[int(i)], float(prob[int(i)])) for i in order]
