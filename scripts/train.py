"""Train a model for fracture classification."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch

from xray_bone_fracture_classifier.config import TrainConfig, make_run_dir, save_config, save_json
from xray_bone_fracture_classifier.utils.seed import set_seed
from xray_bone_fracture_classifier.data.dataloaders import build_dataloaders
from xray_bone_fracture_classifier.models.baseline_cnn import BaselineCNN
from xray_bone_fracture_classifier.models.transfer import build_transfer_model
from xray_bone_fracture_classifier.training.engine import train_model


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train fracture classifier.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--prefix", default="run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        arch=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        label_smoothing=args.label_smoothing,
    )

    set_seed(cfg.seed)

    device = torch.device(cfg.device if cfg.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    loaders, spec = build_dataloaders(Path(cfg.data_dir), cfg.img_size, cfg.batch_size, cfg.num_workers)

    num_classes = len(spec.class_to_idx)
    if cfg.arch == "baseline":
        model = BaselineCNN(num_classes=num_classes)
    else:
        model = build_transfer_model(cfg.arch, num_classes=num_classes, pretrained=True)

    model.to(device)

    run_dir = make_run_dir(Path(args.models_dir), prefix=args.prefix)
    save_config(run_dir, cfg)
    save_json(run_dir / "labels.json", {int(k): v for k, v in spec.idx_to_class.items()})

    model, history = train_model(
        model=model,
        loaders=loaders,
        device=device,
        epochs=cfg.epochs,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        label_smoothing=cfg.label_smoothing,
        early_stopping_patience=cfg.early_stopping_patience,
    )

    torch.save(model.state_dict(), run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "train_history.csv", index=False)

    print("OK. Run saved to:", run_dir)


if __name__ == "__main__":
    main()
