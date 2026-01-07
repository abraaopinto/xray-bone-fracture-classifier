"""Evaluate a trained model on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from xray_bone_fracture_classifier.data.dataloaders import build_dataloaders
from xray_bone_fracture_classifier.inference.predictor import load_bundle
from xray_bone_fracture_classifier.evaluation.metrics import predict_loader, compute_metrics, compute_reports


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate fracture classifier.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--reports-dir", default="reports")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    bundle = load_bundle(model_dir=model_dir)

    loaders, spec = build_dataloaders(
        data_dir=Path(args.data_dir),
        img_size=bundle.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    outputs = predict_loader(bundle.model, loaders["test"], bundle.device)

    target_names = [spec.idx_to_class[i] for i in range(len(spec.idx_to_class))]
    labels = list(range(len(target_names)))
    num_classes = len(target_names)
    metrics = compute_metrics(outputs.y_true, outputs.y_pred, num_classes=num_classes)
    rep, cm = compute_reports(outputs.y_true, outputs.y_pred, target_names=target_names, labels=labels)

    run_id = model_dir.name
    out_dir = Path(args.reports_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(rep).transpose().to_csv(out_dir / "classification_report.csv", index=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix (test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    plt.yticks(range(len(target_names)), target_names)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print("OK. Reports saved to:", out_dir)


if __name__ == "__main__":
    main()
