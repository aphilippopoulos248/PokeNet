"""Step 7 - train.

    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/resnet18.yaml --epochs 40 --batch-size 96

Everything is driven by a YAML config; any key can be overridden on the CLI.
Checkpoints and logs land in outputs/<run_name>/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

from src.dataset import class_names as load_class_names
from src.dataset import make_loaders
from src.engine import (
    EarlyStopping,
    HistoryLogger,
    cosine_warmup,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)
from src.models import build_model, param_groups, set_backbone_trainable
from src.utils import DATA_SPLITS, OUTPUTS, count_params, describe_device, get_device, seed_everything

DEFAULTS = dict(
    run_name="run", model="resnet18", pretrained=True, img_size=224, batch_size=64,
    epochs=30, freeze_epochs=3, lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4,
    optimizer="adamw", label_smoothing=0.1, mixup_alpha=0.0, aug="medium",
    balanced_sampler=False, num_workers=4, patience=10, clip_grad=1.0, amp=True,
    seed=42, width=32, dropout=0.2, warmup_ratio=0.05,
)


def parse_args() -> dict:
    ap = argparse.ArgumentParser(description="Train a Pokemon classifier.")
    ap.add_argument("--config", type=Path)
    for key, val in DEFAULTS.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, dest=key, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, dest=key, type=type(val), default=None)
    ap.add_argument("--splits-dir", type=Path, default=DATA_SPLITS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", type=Path)
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    for key in DEFAULTS:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    cfg["splits_dir"] = args.splits_dir
    cfg["device"] = args.device
    cfg["resume"] = args.resume
    return cfg


def build_optimizer(cfg: dict, model) -> torch.optim.Optimizer:
    groups = param_groups(model, cfg["lr_head"], cfg["lr_backbone"])
    if cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def plot_history(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ep, [r["train_loss"] for r in rows], label="train")
    axes[0].plot(ep, [r["val_loss"] for r in rows], label="val")
    axes[0].set_title("loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, [r["train_top1"] for r in rows], label="train top-1")
    axes[1].plot(ep, [r["val_top1"] for r in rows], label="val top-1")
    axes[1].plot(ep, [r["val_top5"] for r in rows], label="val top-5", ls="--")
    axes[1].set_title("accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    cfg = parse_args()
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    out_dir = OUTPUTS / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = Path(cfg["splits_dir"])
    if not (splits / "train.csv").exists():
        print(f"[train] no manifests in {splits} - run `python -m src.prepare` first")
        return 1
    names = load_class_names(splits)
    num_classes = len(names)

    train_dl, val_dl, test_dl = make_loaders(
        img_size=cfg["img_size"], batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
        aug=cfg["aug"], balanced=cfg["balanced_sampler"], splits_dir=splits,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg["model"], num_classes, cfg["pretrained"], cfg["dropout"], cfg["width"]).to(device)
    total_p, _ = count_params(model)
    print(f"[train] device={describe_device(device)}")
    print(f"[train] model={cfg['model']} params={total_p / 1e6:.1f}M classes={num_classes}")
    print(f"[train] train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if (cfg["amp"] and device.type == "cuda") else None

    freeze_epochs = cfg["freeze_epochs"] if (cfg["pretrained"] and cfg["model"] != "poke_net") else 0
    if freeze_epochs:
        set_backbone_trainable(model, False)
        print(f"[train] phase 1: backbone frozen for {freeze_epochs} epoch(s)")
    optimizer = build_optimizer(cfg, model)

    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = cosine_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)

    history = HistoryLogger(out_dir / "history.csv")
    stopper = EarlyStopping(patience=cfg["patience"])
    best_top1, start_epoch = 0.0, 1

    if cfg["resume"] and Path(cfg["resume"]).exists():
        ck = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state"):
            optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_top1 = ck.get("metrics", {}).get("top1", 0.0)
        print(f"[train] resumed from {cfg['resume']} at epoch {start_epoch}")

    (out_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}, indent=2),
        encoding="utf-8",
    )

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(cfg, model)
            remaining = steps_per_epoch * (cfg["epochs"] - epoch + 1)
            scheduler = cosine_warmup(optimizer, int(remaining * 0.02), remaining)
            print(f"[train] phase 2: backbone unfrozen (lr_backbone={cfg['lr_backbone']:.1e})")

        tr = train_one_epoch(model, train_dl, criterion, optimizer, device, scaler, scheduler,
                             mixup_alpha=cfg["mixup_alpha"], clip_grad=cfg["clip_grad"],
                             desc=f"epoch {epoch}/{cfg['epochs']}")
        va = evaluate(model, val_dl, criterion, device, desc="val")

        row = {"epoch": epoch, "train_loss": tr["loss"], "train_top1": tr["top1"],
               "val_loss": va["loss"], "val_top1": va["top1"], "val_top5": va["top5"],
               "lr": optimizer.param_groups[0]["lr"], "elapsed_s": round(time.time() - t0, 1)}
        history.log(row)
        print(f"epoch {epoch:3d} | train loss {tr['loss']:.3f} top1 {tr['top1']:.3f} "
              f"| val loss {va['loss']:.3f} top1 {va['top1']:.3f} top5 {va['top5']:.3f}")

        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, va, cfg, names)
        if va["top1"] > best_top1:
            best_top1 = va["top1"]
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, va, cfg, names)
            print(f"         new best val top-1 {best_top1:.4f} -> best.pt")

        plot_history(history.rows, out_dir / "curves.png")
        if stopper.step(va["loss"]):
            print(f"[train] early stop at epoch {epoch} (no val-loss improvement for {stopper.patience})")
            break

    print(f"\n[train] done in {(time.time() - t0) / 60:.1f} min | best val top-1 {best_top1:.4f}")
    print(f"[train] artifacts in {out_dir}")
    print(f"[train] next: python -m src.evaluate --checkpoint {out_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
