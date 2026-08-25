"""Step 4 - Dataset, augmentation and DataLoaders."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from src.utils import (
    DATA_SPLITS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROOT,
    load_image_rgb,
    read_json,
)


class PokemonDataset(Dataset):
    """Reads a manifest CSV (path, class, label, split) produced by src.prepare."""

    def __init__(self, manifest: str | Path, transform=None, root: Path = ROOT):
        self.df = pd.read_csv(manifest)
        self.transform = transform
        self.root = Path(root)
        self.labels = self.df["label"].to_numpy()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = load_image_rgb(self.root / row["path"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


def build_transforms(img_size: int = 224, train: bool = False, aug: str = "medium",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Augmentation is the main defence against overfitting on a few-hundred-per-class set.

    Deliberately NO vertical flip and no aggressive hue jitter: Pokemon identity is
    partly colour (Charmander is orange; a hue-shifted Charmander is a different
    creature) and orientation is always upright.
    """
    if not train:
        return T.Compose([
            T.Resize(int(img_size * 1.14)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    scale = {"light": (0.85, 1.0), "medium": (0.7, 1.0), "heavy": (0.5, 1.0)}[aug]
    jitter = {"light": 0.1, "medium": 0.2, "heavy": 0.3}[aug]
    erase = {"light": 0.0, "medium": 0.25, "heavy": 0.4}[aug]

    ops = [
        T.RandomResizedCrop(img_size, scale=scale, ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.RandomRotation(15)], p=0.5),
        T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 4),
    ]
    if aug == "heavy":
        ops.append(T.RandomApply([T.GaussianBlur(3)], p=0.2))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    if erase > 0:
        ops.append(T.RandomErasing(p=erase, scale=(0.02, 0.15)))
    return T.Compose(ops)


def class_names(splits_dir: Path = DATA_SPLITS) -> list[str]:
    return read_json(Path(splits_dir) / "classes.json")["class_names"]


def _balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels)
    weights = 1.0 / np.maximum(counts[labels], 1)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def make_loaders(img_size: int = 224, batch_size: int = 64, num_workers: int = 4,
                 aug: str = "medium", balanced: bool = False,
                 splits_dir: Path = DATA_SPLITS, pin_memory: bool = True):
    splits_dir = Path(splits_dir)
    train_ds = PokemonDataset(splits_dir / "train.csv", build_transforms(img_size, True, aug))
    val_ds = PokemonDataset(splits_dir / "val.csv", build_transforms(img_size, False))
    test_ds = PokemonDataset(splits_dir / "test.csv", build_transforms(img_size, False))

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=num_workers > 0, drop_last=False)
    sampler = _balanced_sampler(train_ds.labels) if balanced else None
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None,
                          sampler=sampler, **common)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_dl = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_dl, val_dl, test_dl


def denormalize(t: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t.cpu() * s + m).clamp(0, 1)
