"""Run inference on an image or directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

from xray_bone_fracture_classifier.inference.predictor import load_bundle, predict_image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Predict fracture class.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main() -> None:
    args = parse_args()
    bundle = load_bundle(Path(args.model_dir))

    if args.image:
        paths = [Path(args.image)]
    elif args.input_dir:
        paths = list(iter_images(Path(args.input_dir)))
    else:
        raise SystemExit("Provide --image or --input-dir")

    rows = []
    for p in paths:
        top = predict_image(bundle=bundle, image_path=p, topk=args.topk)
        pred_label, pred_prob = top[0]
        rows.append({"image": str(p), "pred_label": pred_label, "pred_prob": pred_prob, "topk": str(top)})
        print(p, "=>", top)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image", "pred_label", "pred_prob", "topk"])
            w.writeheader()
            w.writerows(rows)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
