"""
inspect_hbfmid.py

Inspeção do dataset HBFMID (formato YOLO) por split (train/valid/test), gerando:
- reports/hbfmid_split_stats.csv
- reports/hbfmid_summary.json

Métricas coletadas (por split):
- número de imagens
- presença/ausência de labels
- quantidade total de bboxes
- imagens com 0 / 1 / >1 bbox
- contagem de bboxes por classe

Além disso, calcula possíveis colisões de nomes de arquivos entre splits (basenames),
o que pode sinalizar risco de vazamento (leakage) caso a mesma imagem tenha sido
distribuída em splits diferentes.

Requisitos:
- pyyaml
- tqdm

Exemplo:
python scripts/inspect_hbfmid.py --data-dir data/raw/Bone_Fractures_Detection --out-dir reports
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SplitPaths:
    """Estrutura com caminhos esperados para imagens e labels de um split YOLO."""
    split_name: str
    images_dir: Path
    labels_dir: Path


@dataclass
class SplitStats:
    """Estatísticas agregadas por split."""
    split: str
    num_images: int = 0
    num_missing_label_file: int = 0
    num_empty_label_file: int = 0
    num_images_0_bbox: int = 0
    num_images_1_bbox: int = 0
    num_images_multi_bbox: int = 0
    num_total_boxes: int = 0


def parse_args() -> argparse.Namespace:
    """Parse de argumentos CLI."""
    ap = argparse.ArgumentParser(description="Inspeciona dataset YOLO do HBFMID e gera relatórios.")
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Diretório raiz do dataset (contém data.yaml e pastas train/valid/test).",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Diretório de saída para relatórios (CSV/JSON).",
    )
    ap.add_argument(
        "--splits",
        default="train,valid,test",
        help="Lista de splits separados por vírgula (default: train,valid,test).",
    )
    ap.add_argument(
        "--check-cross-split-basename-collisions",
        action="store_true",
        help="Verifica colisões de basenames (nomes de arquivos) entre splits (sinal de leakage).",
    )
    return ap.parse_args()


def load_class_names(data_yaml_path: Path) -> List[str]:
    """
    Carrega nomes de classes a partir do data.yaml (YOLO).

    Suporta formatos comuns:
    - names: ["a", "b", ...]
    - names: {0: "a", 1: "b", ...}

    Args:
        data_yaml_path: Caminho para data.yaml.

    Returns:
        Lista de nomes de classes, indexada pelo id da classe.
    """
    with data_yaml_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict) or "names" not in payload:
        raise ValueError(f"Estrutura inesperada em {data_yaml_path}. Esperado dict com chave 'names'.")

    names = payload["names"]
    if isinstance(names, list):
        return [str(x) for x in names]

    if isinstance(names, dict):
        # Garantir ordenação por id
        out: List[str] = []
        for k in sorted(names.keys(), key=lambda x: int(x)):
            out.append(str(names[k]))
        return out

    raise ValueError(f"Campo 'names' em {data_yaml_path} está em formato não suportado: {type(names)}")


def resolve_split_paths(data_dir: Path, split_name: str) -> SplitPaths:
    """
    Resolve os diretórios de imagens e labels para um split.

    Tenta padrões comuns:
    - {split}/images e {split}/labels
    - {split} (imagens direto) e {split}/labels
    - {split}/images e {split} (labels direto)  [menos comum]
    """
    split_root = data_dir / split_name

    # Padrão YOLO clássico
    images_dir = split_root / "images"
    labels_dir = split_root / "labels"

    if images_dir.exists() and labels_dir.exists():
        return SplitPaths(split_name, images_dir, labels_dir)

    # Alternativas: imagens direto no split
    if split_root.exists():
        # imagens no split_root e labels em split_root/labels
        if labels_dir.exists():
            return SplitPaths(split_name, split_root, labels_dir)

        # imagens em split_root/images e labels direto no split_root
        if images_dir.exists():
            return SplitPaths(split_name, images_dir, split_root)

        # ambos direto no split_root (diferenciar por extensão .txt)
        return SplitPaths(split_name, split_root, split_root)

    raise FileNotFoundError(f"Split '{split_name}' não encontrado em: {split_root}")


def iter_images(images_dir: Path) -> Iterable[Path]:
    """Itera arquivos de imagem no diretório (não recursivo) filtrando por extensão."""
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def parse_yolo_label_file(label_path: Path) -> Tuple[List[int], int]:
    """
    Parse de um arquivo YOLO label (.txt).

    Formato esperado (por linha): class_id x_center y_center width height [extras...]

    Args:
        label_path: Caminho do arquivo .txt.

    Returns:
        (class_ids, num_boxes)
    """
    content = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return [], 0

    class_ids: List[int] = []
    n = 0
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            # linha inválida -> ignora
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        class_ids.append(cid)
        n += 1

    return class_ids, n


def compute_stats_for_split(
    split_paths: SplitPaths,
    class_names: Sequence[str],
) -> Tuple[SplitStats, List[int], List[str]]:
    """
    Calcula estatísticas do split e retorna contagens por classe.

    Args:
        split_paths: Caminhos resolvidos (imagens/labels).
        class_names: Lista de nomes de classes.

    Returns:
        (stats, class_counts, image_basenames)
        - class_counts: contagem de bboxes por classe (len == num_classes)
        - image_basenames: lista de basenames das imagens do split (para checagem cross-split)
    """
    stats = SplitStats(split=split_paths.split_name)
    class_counts = [0] * len(class_names)
    basenames: List[str] = []

    images = sorted(iter_images(split_paths.images_dir))
    for img_path in tqdm(images, desc=f"[{split_paths.split_name}]", unit="img"):
        stats.num_images += 1
        basenames.append(img_path.name)

        label_path = split_paths.labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats.num_missing_label_file += 1
            stats.num_images_0_bbox += 1
            continue

        class_ids, n_boxes = parse_yolo_label_file(label_path)
        if n_boxes == 0:
            stats.num_empty_label_file += 1
            stats.num_images_0_bbox += 1
            continue

        stats.num_total_boxes += n_boxes
        if n_boxes == 1:
            stats.num_images_1_bbox += 1
        else:
            stats.num_images_multi_bbox += 1

        for cid in class_ids:
            if 0 <= cid < len(class_counts):
                class_counts[cid] += 1

    return stats, class_counts, basenames


def write_split_stats_csv(
    csv_path: Path,
    split_stats: List[SplitStats],
    class_names: Sequence[str],
    split_class_counts: Dict[str, List[int]],
) -> None:
    """
    Escreve CSV consolidado com estatísticas por split.

    Args:
        csv_path: Saída do CSV.
        split_stats: Lista com stats por split.
        class_names: nomes de classes.
        split_class_counts: dict split -> counts por classe.
    """
    header = [
        "split",
        "num_images",
        "num_missing_label_file",
        "num_empty_label_file",
        "num_images_0_bbox",
        "num_images_1_bbox",
        "num_images_multi_bbox",
        "num_total_boxes",
    ]
    header += [f"bbox_count__{name}" for name in class_names]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for s in split_stats:
            row = [
                s.split,
                s.num_images,
                s.num_missing_label_file,
                s.num_empty_label_file,
                s.num_images_0_bbox,
                s.num_images_1_bbox,
                s.num_images_multi_bbox,
                s.num_total_boxes,
            ]
            row += split_class_counts.get(s.split, [0] * len(class_names))
            writer.writerow(row)


def cross_split_basename_collisions(split_basenames: Dict[str, List[str]]) -> Dict[str, object]:
    """
    Verifica interseção de basenames entre splits.

    Args:
        split_basenames: dict split -> lista de basenames das imagens.

    Returns:
        Um dicionário com pares de splits e quantidade/itens em interseção (limitado).
    """
    splits = sorted(split_basenames.keys())
    sets = {s: set(split_basenames[s]) for s in splits}

    collisions: Dict[str, object] = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            inter = sorted(sets[a].intersection(sets[b]))
            key = f"{a}__x__{b}"
            collisions[key] = {
                "count": len(inter),
                "examples": inter[:50],  # evita JSON gigante
            }
    return collisions


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = data_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml não encontrado em: {data_yaml}")

    class_names = load_class_names(data_yaml)
    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    split_stats_list: List[SplitStats] = []
    split_class_counts: Dict[str, List[int]] = {}
    split_basenames: Dict[str, List[str]] = {}

    total_class_counts = [0] * len(class_names)
    total_boxes = 0

    for split in requested_splits:
        sp = resolve_split_paths(data_dir, split)
        stats, class_counts, basenames = compute_stats_for_split(sp, class_names)

        split_stats_list.append(stats)
        split_class_counts[split] = class_counts
        split_basenames[split] = basenames

        total_boxes += stats.num_total_boxes
        total_class_counts = [a + b for a, b in zip(total_class_counts, class_counts)]

    # CSV por split
    csv_path = out_dir / "hbfmid_split_stats.csv"
    write_split_stats_csv(csv_path, split_stats_list, class_names, split_class_counts)

    # JSON summary
    summary: Dict[str, object] = {
        "data_dir": str(data_dir),
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "splits": {},
        "bbox_class_counts_total": {class_names[i]: total_class_counts[i] for i in range(len(class_names))},
        "num_total_boxes": total_boxes,
    }

    for stats in split_stats_list:
        split = stats.split
        summary["splits"][split] = {
            "num_images": stats.num_images,
            "num_missing_label_file": stats.num_missing_label_file,
            "num_empty_label_file": stats.num_empty_label_file,
            "num_images_0_bbox": stats.num_images_0_bbox,
            "num_images_1_bbox": stats.num_images_1_bbox,
            "num_images_multi_bbox": stats.num_images_multi_bbox,
            "num_total_boxes": stats.num_total_boxes,
            "bbox_class_counts": {class_names[i]: split_class_counts[split][i] for i in range(len(class_names))},
        }

    if args.check_cross_split_basename_collisions:
        summary["cross_split_basename_collisions"] = cross_split_basename_collisions(split_basenames)

    (out_dir / "hbfmid_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("OK. Relatórios gerados em:")
    print(f"- {csv_path}")
    print(f"- {out_dir / 'hbfmid_summary.json'}")


if __name__ == "__main__":
    main()