"""Data loaders for ImageFolder datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataSpec:
    class_to_idx: Dict[str, int]
    idx_to_class: Dict[int, str]


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Build train and eval transforms."""
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms

def build_dataloaders(
    data_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], DataSpec]:
    """Create DataLoaders for train/valid/test using ImageFolder,
    enforcing a consistent class_to_idx across splits.
    """
    train_tfms, eval_tfms = build_transforms(img_size)

    train_ds = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_tfms)
    valid_ds = datasets.ImageFolder(root=str(data_dir / "valid"), transform=eval_tfms)
    test_ds  = datasets.ImageFolder(root=str(data_dir / "test"),  transform=eval_tfms)

    # Canonical mapping from train
    class_to_idx = dict(train_ds.class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    def _remap_split_targets(split_ds: datasets.ImageFolder) -> None:
        """
        Remap split targets to train's class_to_idx, based on folder (class) name.
        This prevents index mismatch when classes are missing or ordered differently.
        """
        split_idx_to_class = {v: k for k, v in split_ds.class_to_idx.items()}

        new_samples = []
        for path, split_target in split_ds.samples:
            class_name = split_idx_to_class[int(split_target)]
            if class_name not in class_to_idx:
                raise ValueError(
                    f"Class '{class_name}' found in split but not in train. "
                    f"Check folder structure under {split_ds.root}."
                )
            new_target = class_to_idx[class_name]
            new_samples.append((path, new_target))

        split_ds.samples = new_samples
        split_ds.targets = [t for _, t in new_samples]

        # also overwrite mappings for consistency
        split_ds.class_to_idx = class_to_idx
        split_ds.classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Remap valid and test to train mapping
    _remap_split_targets(valid_ds)
    _remap_split_targets(test_ds)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "valid": DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }
    return loaders, DataSpec(class_to_idx=class_to_idx, idx_to_class=idx_to_class)
