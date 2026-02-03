from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from xray_bone_fracture_classifier.data.dataloaders import IMAGENET_MEAN, IMAGENET_STD
from xray_bone_fracture_classifier.inference.predictor import ModelBundle, predict_image
from xray_bone_fracture_classifier.models.transfer import build_transfer_model


def test_predict_image_smoke(tmp_path: Path) -> None:
    # dummy image on disk
    img = Image.fromarray((np.random.rand(224, 224) * 255).astype(np.uint8), mode="L")
    img_path = tmp_path / "xray.png"
    img.save(img_path)

    idx_to_class = {0: "no_fracture", 1: "fracture"}
    class_to_idx = {"no_fracture": 0, "fracture": 1}

    model = build_transfer_model(arch="resnet18", num_classes=2, pretrained=False)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    bundle = ModelBundle(
        model=model,
        idx_to_class=idx_to_class,
        class_to_idx=class_to_idx,
        transform=transform,
        img_size=224,
        device=torch.device("cpu"),
    )

    out = predict_image(bundle=bundle, image_path=img_path, topk=2)

    assert isinstance(out, list)
    assert len(out) == 2
    for label, prob in out:
        assert label in {"no_fracture", "fracture"}
        assert 0.0 <= prob <= 1.0
