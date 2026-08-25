"""Shared helpers: paths, seeding, device selection, image loading."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

# Some Kaggle images are truncated; let PIL load them rather than crash mid-epoch.
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_SPLITS = ROOT / "data" / "splits"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif"}

# ImageNet statistics - required when using pretrained torchvision weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Make a run reproducible. `deterministic=True` costs speed on GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        i = device.index or 0
        name = torch.cuda.get_device_name(i)
        gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
        return f"cuda:{i} ({name}, {gb:.1f} GB)"
    return str(device)


def load_image_rgb(path: str | Path, bg: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Open an image as RGB.

    Pokemon sprite datasets are full of PNGs with an alpha channel. Calling
    .convert("RGB") on those makes transparent pixels BLACK, which invents a
    hard black silhouette the model will happily learn instead of the Pokemon.
    We composite onto a solid background instead.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, bg + (255,))
        img = Image.alpha_composite(canvas, img)
    return img.convert("RGB")


def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
