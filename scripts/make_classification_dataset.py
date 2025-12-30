"""
make_classification_dataset.py

Constrói um dataset de CLASSIFICAÇÃO a partir de um dataset YOLO (detecção),
gerando crops das bounding boxes.

Modos:
- bbox: gera um crop por bbox (um arquivo por bbox)
- largest: gera um crop apenas da maior bbox por imagem

Saída:
- out_dir/{train,valid,test}/{class_name}/*.jpg
- out_dir/manifest.csv (linhagem e metadados por crop)
- out_dir/build_stats.txt (resumo da construção)

Requisitos:
- pyyaml
- pillow
- tqdm

Exemplos:
python scripts/make_classification_dataset.py --data-dir data/raw/Bone_Fractures_Detection --out-dir data/processed/hbfmid_cls_bbox --mode bbox
python scripts/make_classification_dataset.py --data-dir data/raw/Bone_Fractures_Detection --out-dir data/processed/hbfmid_cls_largest --mode largest
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from PIL import Image
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class YoloBox:
    """Bounding box YOLO normalizada (xc, yc, w, h) em [0,1] e class_id."""
    class_id: int
    xc: float
    yc: float
    w: float
    h: float


def parse_args() -> argparse.Namespace:
    """Parse de argumentos CLI."""
    ap = argparse.ArgumentParser(description="Converte dataset YOLO (detecção) em dataset de classificação (crops).")
    ap.add_argument("--data-dir", required=True, help="Diretório raiz do dataset (contém data.yaml e splits).")
    ap.add_argument("--out-dir", required=True, help="Diretório de saída do dataset de classificação.")
    ap.add_argument(
        "--mode",
        choices=("bbox", "largest"),
        required=True,
        help="bbox = um crop por bbox; largest = apenas a maior bbox por imagem.",
    )
    ap.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Padding relativo aplicado à bbox (ex: 0.10 adiciona 10%%). Default=0.",
    )
    ap.add_argument(
        "--min-area",
        type=int,
        default=1,
        help="Área mínima do crop em pixels para ser salvo. Default=1.",
    )
    ap.add_argument(
        "--splits",
        default="train,valid,test",
        help="Lista de splits separados por vírgula (default: train,valid,test).",
    )
    ap.add_argument(
        "--image-format",
        choices=("jpg", "png"),
        default="jpg",
        help="Formato de salvamento das imagens (default: jpg).",
    )
    ap.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="Qualidade JPEG (apenas se image-format=jpg). Default=95.",
    )
    ap.add_argument(
        "--skip-images-without-labels",
        action="store_true",
        help="Se ativado, ignora imagens sem arquivo .txt de label.",
    )
    ap.add_argument(
        "--skip-empty-label-files",
        action="store_true",
        help="Se ativado, ignora labels vazios (sem bbox).",
    )
    return ap.parse_args()


def load_class_names(data_yaml_path: Path) -> List[str]:
    """
    Carrega nomes de classes do data.yaml.

    Args:
        data_yaml_path: Caminho do data.yaml.

    Returns:
        Lista de nomes de classes.
    """
    with data_yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict) or "names" not in payload:
        raise ValueError(f"Estrutura inesperada em {data_yaml_path}. Esperado dict com chave 'names'.")

    names = payload["names"]
    if isinstance(names, list):
        return [str(x) for x in names]

    if isinstance(names, dict):
        out: List[str] = []
        for k in sorted(names.keys(), key=lambda x: int(x)):
            out.append(str(names[k]))
        return out

    raise ValueError(f"Campo 'names' em {data_yaml_path} está em formato não suportado: {type(names)}")


def ensure_dir(path: Path) -> None:
    """Cria diretório se não existir."""
    path.mkdir(parents=True, exist_ok=True)


def iter_images(images_dir: Path) -> List[Path]:
    """Lista imagens (não recursivo) no diretório por extensão suportada."""
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def resolve_split_dirs(data_dir: Path, split_name: str) -> Tuple[Path, Path]:
    """
    Resolve dirs de imagens e labels para um split.

    Retorna:
        (images_dir, labels_dir)
    """
    split_root = data_dir / split_name
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"

    if images_dir.exists() and labels_dir.exists():
        return images_dir, labels_dir

    if split_root.exists():
        if labels_dir.exists():
            return split_root, labels_dir
        if images_dir.exists():
            return images_dir, split_root
        return split_root, split_root

    raise FileNotFoundError(f"Split '{split_name}' não encontrado em: {split_root}")


def parse_yolo_labels(label_path: Path) -> List[YoloBox]:
    """
    Parse de labels YOLO.

    Formato esperado:
        class_id x_center y_center width height [extras...]

    Args:
        label_path: Caminho do arquivo label.

    Returns:
        Lista de YoloBox.
    """
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    boxes: List[YoloBox] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        boxes.append(YoloBox(class_id=class_id, xc=xc, yc=yc, w=w, h=h))
    return boxes


def yolo_to_xyxy_pixels(
    box: YoloBox,
    img_w: int,
    img_h: int,
    padding: float = 0.0,
) -> Tuple[int, int, int, int]:
    """
    Converte bbox YOLO normalizada para coordenadas pixel (x1,y1,x2,y2), com padding e clipping.

    Args:
        box: bbox YOLO.
        img_w: largura da imagem.
        img_h: altura da imagem.
        padding: fator relativo (ex.: 0.10 = +10% em cada lado, proporcional ao tamanho da bbox).

    Returns:
        (x1, y1, x2, y2) em pixels, onde x2/y2 são exclusivos (padrão PIL crop).
    """
    xc = box.xc * img_w
    yc = box.yc * img_h
    bw = box.w * img_w
    bh = box.h * img_h

    pad_w = bw * padding
    pad_h = bh * padding

    x1 = int(round(xc - bw / 2 - pad_w))
    y1 = int(round(yc - bh / 2 - pad_h))
    x2 = int(round(xc + bw / 2 + pad_w))
    y2 = int(round(yc + bh / 2 + pad_h))

    # clipping
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2


def choose_largest_box(boxes: Sequence[YoloBox]) -> Optional[Tuple[int, YoloBox]]:
    """
    Seleciona a bbox de maior área (normalizada), retornando (index, box).

    Args:
        boxes: lista de bboxes.

    Returns:
        (idx, box) ou None se lista vazia.
    """
    if not boxes:
        return None

    best_idx = -1
    best_area = -1.0
    best_box: Optional[YoloBox] = None

    for i, b in enumerate(boxes):
        area = float(b.w) * float(b.h)
        if area > best_area:
            best_area = area
            best_idx = i
            best_box = b

    if best_box is None:
        return None
    return best_idx, best_box


def save_crop(
    crop: Image.Image,
    out_path: Path,
    image_format: str,
    jpg_quality: int,
) -> None:
    """Salva crop em disco com parâmetros apropriados."""
    ensure_dir(out_path.parent)

    if image_format == "jpg":
        crop.convert("RGB").save(out_path, format="JPEG", quality=jpg_quality, optimize=True)
    else:
        crop.convert("RGB").save(out_path, format="PNG", optimize=True)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado: {data_yaml}")

    class_names = load_class_names(data_yaml)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    manifest_path = out_dir / "manifest.csv"
    stats_path = out_dir / "build_stats.txt"

    # Estatísticas agregadas
    stats: Dict[str, int] = {
        "num_images_seen": 0,
        "num_images_missing_label_file": 0,
        "num_images_empty_label_file": 0,
        "num_boxes_seen": 0,
        "num_crops_saved": 0,
        "num_crops_skipped_too_small": 0,
        "num_crops_skipped_invalid_class": 0,
    }
    crops_per_split: Dict[str, int] = {s: 0 for s in splits}
    crops_per_class: Dict[str, int] = {name: 0 for name in class_names}

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "split",
                "class_id",
                "class_name",
                "src_image",
                "src_label",
                "dst_crop",
                "bbox_index",
                "x1",
                "y1",
                "x2",
                "y2",
                "img_w",
                "img_h",
                "mode",
                "padding",
            ]
        )

        for split in splits:
            images_dir, labels_dir = resolve_split_dirs(data_dir, split)
            images = iter_images(images_dir)

            for img_path in tqdm(images, desc=f"[build:{split}]", unit="img"):
                stats["num_images_seen"] += 1

                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    stats["num_images_missing_label_file"] += 1
                    if args.skip_images_withtout_labels if False else False:
                        pass  # placeholder (mantém compatibilidade; ver abaixo)
                    # compat: arg real é --skip-images-without-labels
                    if args.skip_images_without_labels:
                        continue
                    boxes: List[YoloBox] = []
                else:
                    boxes = parse_yolo_labels(label_path)
                    if not boxes:
                        stats["num_images_empty_label_file"] += 1
                        if args.skip_empty_label_files:
                            continue

                # carregar imagem
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size

                    if args.mode == "largest":
                        sel = choose_largest_box(boxes)
                        if sel is None:
                            continue
                        bbox_indices = [sel[0]]
                        selected_boxes = [sel[1]]
                    else:
                        bbox_indices = list(range(len(boxes)))
                        selected_boxes = list(boxes)

                    for bbox_idx, box in zip(bbox_indices, selected_boxes):
                        stats["num_boxes_seen"] += 1

                        if not (0 <= box.class_id < len(class_names)):
                            stats["num_crops_skipped_invalid_class"] += 1
                            continue

                        x1, y1, x2, y2 = yolo_to_xyxy_pixels(
                            box=box,
                            img_w=img_w,
                            img_h=img_h,
                            padding=args.padding,
                        )

                        # valida área
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if area < args.min_area or x2 <= x1 or y2 <= y1:
                            stats["num_crops_skipped_too_small"] += 1
                            continue

                        crop = img.crop((x1, y1, x2, y2))

                        class_name = class_names[box.class_id]
                        out_name = f"{img_path.stem}__bbox{bbox_idx}.{args.image_format}"
                        dst_crop = out_dir / split / class_name / out_name

                        save_crop(
                            crop=crop,
                            out_path=dst_crop,
                            image_format=args.image_format,
                            jpg_quality=args.jpg_quality,
                        )

                        stats["num_crops_saved"] += 1
                        crops_per_split[split] += 1
                        crops_per_class[class_name] += 1

                        writer.writerow(
                            [
                                split,
                                box.class_id,
                                class_name,
                                str(img_path.resolve()),
                                str(label_path.resolve()) if label_path.exists() else "",
                                str(dst_crop.resolve()),
                                bbox_idx,
                                x1,
                                y1,
                                x2,
                                y2,
                                img_w,
                                img_h,
                                args.mode,
                                args.padding,
                            ]
                        )

    # build_stats.txt
    lines: List[str] = []
    lines.append("HBFMID - Build Classification Dataset")
    lines.append("")
    lines.append("Parameters")
    lines.append(f"- data_dir: {data_dir}")
    lines.append(f"- out_dir: {out_dir}")
    lines.append(f"- mode: {args.mode}")
    lines.append(f"- padding: {args.padding}")
    lines.append(f"- min_area: {args.min_area}")
    lines.append(f"- image_format: {args.image_format}")
    lines.append(f"- jpg_quality: {args.jpg_quality}")
    lines.append(f"- splits: {splits}")
    lines.append(f"- skip_images_without_labels: {args.skip_images_without_labels}")
    lines.append(f"- skip_empty_label_files: {args.skip_empty_label_files}")
    lines.append("")
    lines.append("Global Stats")
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Crops per split")
    for s in splits:
        lines.append(f"- {s}: {crops_per_split[s]}")
    lines.append("")
    lines.append("Crops per class")
    for cname in class_names:
        lines.append(f"- {cname}: {crops_per_class[cname]}")

    stats_path.write_text("\n".join(lines), encoding="utf-8")

    print("OK. Dataset de classificação gerado em:")
    print(f"- {out_dir}")
    print("Artefatos:")
    print(f"- {manifest_path}")
    print(f"- {stats_path}")


if __name__ == "__main__":
    main()