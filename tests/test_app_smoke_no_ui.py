from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from xray_bone_fracture_classifier.inference.predictor import load_bundle, predict_image


def test_app_core_flow_smoke(tmp_path: Path) -> None:
    models_root = Path("models")
    runs = sorted([p for p in models_root.iterdir() if p.is_dir()], reverse=True)

    # filtra runs válidos
    run = None
    for p in runs:
        if (p / "model.pt").exists() and (p / "config.json").exists() and (p / "labels.json").exists():
            run = p
            break

    assert run is not None, "No valid model run found in ./models"

    bundle = load_bundle(model_dir=run, device="cpu")

    # imagem dummy
    img = Image.fromarray((np.random.rand(224, 224) * 255).astype(np.uint8), mode="L").convert("RGB")
    img_path = tmp_path / "xray.png"
    img.save(img_path)

    preds = predict_image(bundle=bundle, image_path=img_path, topk=3)
    assert isinstance(preds, list)
    assert len(preds) >= 1
    assert isinstance(preds[0][0], str)
    assert 0.0 <= float(preds[0][1]) <= 1.0
