"""
Download do dataset HBFMID via kagglehub.

Este script baixa o dataset 'human-bone-fractures-image-dataset' do Kaggle
e copia/organiza a estrutura para o diretório data/raw/ do projeto, sem
versionar os dados no Git.

Requisitos:
- kagglehub instalado
- credenciais Kaggle configuradas no ambiente (KAGGLE_USERNAME/KAGGLE_KEY)
  ou arquivo kaggle.json conforme documentação do Kaggle.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub


def _find_expected_root(download_path: Path) -> Path:
    """
    Tenta localizar o diretório raiz do dataset dentro do cache do kagglehub.

    O kagglehub retorna um caminho de cache; dependendo da versão/estrutura,
    o conteúdo pode estar nesse diretório ou em subpastas.

    Retorna o diretório que contém 'data.yaml' ou 'Bone_Fractures_Detection'.
    """
    candidates = [download_path] + list(download_path.rglob("*"))
    for p in candidates:
        if not p.is_dir():
            continue
        if (p / "data.yaml").exists() or (p / "Bone_Fractures_Detection").exists():
            return p
    return download_path


def download_dataset(out_dir: Path, force: bool = False) -> Path:
    """
    Baixa o dataset do Kaggle (via kagglehub) e copia para out_dir.

    Args:
        out_dir: Diretório destino (ex: data/raw)
        force: Se True, remove o destino antes de copiar.

    Returns:
        Caminho do dataset copiado no out_dir.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cache_path = Path(
        kagglehub.dataset_download("jockeroika/human-bone-fractures-image-dataset")
    ).resolve()

    root = _find_expected_root(dataset_cache_path)

    # Você pode ajustar esse nome conforme seu padrão interno.
    target = out_dir / "Bone_Fractures_Detection"

    if target.exists():
        if not force:
            return target
        shutil.rmtree(target)

    # Se o dataset já vier com Bone_Fractures_Detection dentro do root, copiamos a pasta.
    source = root / "Bone_Fractures_Detection"
    if source.exists():
        shutil.copytree(source, target)
        return target

    # Caso a estrutura seja diferente, copiamos tudo para target
    shutil.copytree(root, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa o dataset HBFMID via kagglehub.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/raw",
        help="Diretório destino (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força re-download/cópia removendo o destino existente.",
    )

    args = parser.parse_args()
    target = download_dataset(Path(args.out_dir), force=args.force)
    print(f"[OK] Dataset disponível em: {target}")


if __name__ == "__main__":
    main()