"""Answer 'how long will this actually take?' on YOUR GPU, not on a forum post.

Times the data loader alone and a full training step, then extrapolates to an
epoch and a whole run. Works before you have any data (--synthetic).

    python -m src.benchmark --config configs/resnet18.yaml
    python -m src.benchmark --config configs/resnet50.yaml --synthetic
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.dataset import make_loaders
from src.models import build_model
from src.utils import DATA_SPLITS, describe_device, get_device, read_json


def fmt(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_loader(loader, steps: int) -> tuple[float, int]:
    it = iter(loader)
    images, _ = next(it)  # warm the workers
    bs = images.size(0)
    t0 = time.perf_counter()
    n = 0
    for _ in range(steps):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        n += images.size(0)
    return time.perf_counter() - t0, n or bs


def time_train_step(model, device, batch, steps: int, amp: bool) -> float:
    images, targets = batch
    images, targets = images.to(device), targets.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda") if (amp and device.type == "cuda") else None

    for _ in range(3):  # warmup: cudnn autotune + lazy init
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images), targets)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    sync(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            loss = criterion(model(images), targets)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
    sync(device)
    return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--synthetic", action="store_true", help="skip real data, use random tensors")
    ap.add_argument("--train-images", type=int, help="override the training-set size estimate")
    ap.add_argument("--splits-dir", type=Path, default=DATA_SPLITS)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = get_device(args.device)
    bs, size = cfg["batch_size"], cfg["img_size"]

    have_data = (args.splits_dir / "train.csv").exists() and not args.synthetic
    num_classes = len(read_json(args.splits_dir / "classes.json")["class_names"]) if \
        (args.splits_dir / "classes.json").exists() else 151

    print(f"\ndevice     {describe_device(device)}")
    print(f"config     {args.config.name}  ({cfg['model']}, {size}px, batch {bs}, amp={cfg['amp']})")

    loader_imgs_per_s = None
    if have_data:
        train_dl, _, _ = make_loaders(size, bs, cfg["num_workers"], cfg["aug"],
                                      cfg["balanced_sampler"], args.splits_dir,
                                      pin_memory=device.type == "cuda")
        n_train = len(train_dl.dataset)
        elapsed, n = time_loader(train_dl, args.steps)
        loader_imgs_per_s = n / elapsed
        batch = next(iter(train_dl))
        print(f"data       {n_train} training images")
        print(f"loader     {loader_imgs_per_s:>7.0f} img/s  ({cfg['num_workers']} workers, decode + augment)")
    else:
        n_train = args.train_images or 5000
        batch = (torch.randn(bs, 3, size, size), torch.randint(0, num_classes, (bs,)))
        print(f"data       synthetic, assuming {n_train} training images")

    model = build_model(cfg["model"], num_classes, cfg["pretrained"],
                        cfg["dropout"], cfg.get("width", 32)).to(device)
    elapsed = time_train_step(model, device, batch, args.steps, cfg["amp"])
    step_imgs_per_s = args.steps * bs / elapsed
    print(f"gpu step   {step_imgs_per_s:>7.0f} img/s  (forward + backward, data already resident)")

    # The slower of the two is what you actually get.
    effective = min(step_imgs_per_s, loader_imgs_per_s) if loader_imgs_per_s else step_imgs_per_s
    bound = "data loader" if (loader_imgs_per_s and loader_imgs_per_s < step_imgs_per_s) else "GPU"

    epoch_s = n_train / effective
    val_s = epoch_s * 0.12  # eval is forward-only on a smaller set
    total_s = (epoch_s + val_s) * cfg["epochs"]

    print(f"\nbottleneck {bound}")
    print(f"per epoch  ~{fmt(epoch_s + val_s)}  (train {fmt(epoch_s)} + val {fmt(val_s)})")
    print(f"full run   ~{fmt(total_s)}  ({cfg['epochs']} epochs, before early stopping)")

    if bound == "data loader":
        print("\nThe GPU is idling. Raise num_workers, or drop img_size to 160.")
    if not have_data:
        print("\nSynthetic estimate - loader cost is not included. Re-run after src.prepare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
