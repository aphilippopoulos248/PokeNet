"""Step 7 - train.

    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/resnet18.yaml --epochs 40 --batch-size 96

    # continue a finished run for 60 more epochs (--epochs is the new TOTAL,
    # not "epochs to add" - 60 already done + 60 more = 120):
    python -m src.train --config configs/baseline.yaml --epochs 120 \
        --resume outputs/baseline_pokenet/best.pt

Everything is driven by a YAML config; any key can be overridden on the CLI.
Checkpoints and logs land in outputs/<run_name>/.

This is the file that ties everything else together: src/dataset.py builds the
data pipeline, src/models.py builds the network, src/engine.py runs the actual
training/eval math - main() below just calls each of those, in order, once per
epoch, and handles the surrounding bookkeeping (config, checkpoints, logging,
the freeze/unfreeze phase switch, early stopping).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend - required so plotting works with no display attached (e.g. over SSH, in a script)
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

# The single source of truth for every tunable knob and its default value.
# Three things read this same dict: the CLI (parse_args builds one --flag per
# key below), a YAML config file (which can override any subset of these
# keys), and the training loop itself (cfg["key"] everywhere in main()).
# Precedence, highest to lowest: CLI flag > YAML config > this default.
DEFAULTS = dict(
    run_name="run", model="resnet18", pretrained=True, img_size=224, batch_size=64,
    epochs=30, freeze_epochs=3, lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4,
    optimizer="adamw", label_smoothing=0.1, mixup_alpha=0.0, aug="medium",
    balanced_sampler=False, num_workers=4, patience=10, clip_grad=1.0, amp=True,
    seed=42, width=32, dropout=0.2, warmup_ratio=0.05,
)


def parse_args() -> dict:
    """Merge CLI flags, an optional --config YAML file, and DEFAULTS into one settled config dict."""
    ap = argparse.ArgumentParser(description="Train a Pokemon classifier.")
    ap.add_argument("--config", type=Path)
    # Auto-generates one CLI flag per DEFAULTS entry (run_name -> --run-name,
    # lr_head -> --lr-head, ...) instead of hand-writing 20 near-identical
    # add_argument calls. Each flag defaults to None here specifically -
    # `None` means "the user didn't pass this flag", which is what lets the
    # merge loop below tell "explicitly set on the CLI" apart from "just
    # happens to match the default value".
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

    # Layer 1: start from the hard-coded defaults.
    cfg = dict(DEFAULTS)
    # Layer 2: a --config file overrides whichever keys it mentions (e.g.
    # configs/resnet18.yaml sets model, freeze_epochs, lr_backbone, ...) and
    # leaves anything it doesn't mention at the DEFAULTS value.
    if args.config:
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {})
    # Layer 3: an explicit CLI flag beats everything - this is what makes
    # `--epochs 40` on the command line override whatever configs/*.yaml said,
    # which is exactly how the "resume for 60 more epochs" workflow overrides
    # a config's baked-in epoch count.
    for key in DEFAULTS:
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    cfg["splits_dir"] = args.splits_dir
    cfg["device"] = args.device
    cfg["resume"] = args.resume
    return cfg


def build_optimizer(cfg: dict, model) -> torch.optim.Optimizer:
    """AdamW is the default (adapts its own per-parameter learning rate, forgiving of
    imperfect LR choices); SGD+momentum is offered as the classic, more
    hand-tuned alternative some configs may prefer.
    """
    groups = param_groups(model, cfg["lr_head"], cfg["lr_backbone"])
    if cfg["optimizer"].lower() == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"])
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def plot_history(rows: list[dict], out: Path) -> None:
    """Redraws outputs/<run_name>/curves.png from scratch after every epoch.

    Two panels side by side: loss (train vs val - the gap between them is the
    quickest way to spot overfitting) and accuracy (train/val top-1 plus val
    top-5, so you can see all three trends at a glance without re-running
    anything).
    """
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
    plt.close(fig)  # frees the figure's memory - matters over a long run redrawing every epoch


def main() -> int:
    # ---- 1. settle the config, make the run reproducible -------------------
    cfg = parse_args()
    seed_everything(cfg["seed"])  # same seed -> same shuffle order, same augmentation rolls, same init
    device = get_device(cfg["device"])
    out_dir = OUTPUTS / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. data -------------------------------------------------------------
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

    # ---- 3. model --------------------------------------------------------------
    model = build_model(cfg["model"], num_classes, cfg["pretrained"], cfg["dropout"], cfg["width"]).to(device)
    total_p, _ = count_params(model)
    print(f"[train] device={describe_device(device)}")
    print(f"[train] model={cfg['model']} params={total_p / 1e6:.1f}M classes={num_classes}")
    print(f"[train] train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)}")

    # label_smoothing softens the training targets slightly (e.g. 0.9/0.1
    # spread across the wrong classes instead of a hard 1.0/0.0), which
    # discourages the model from becoming overconfident on a training set this
    # size. GradScaler is only created for CUDA + amp=True - see engine.py's
    # train_one_epoch for what it actually does.
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = torch.amp.GradScaler("cuda") if (cfg["amp"] and device.type == "cuda") else None

    # ---- 4. transfer-learning phase 1: freeze the pretrained backbone --------
    # poke_net has no pretrained backbone to freeze (pretrained=False always
    # for it), so freeze_epochs is forced to 0 in that case regardless of what
    # a config says.
    freeze_epochs = cfg["freeze_epochs"] if (cfg["pretrained"] and cfg["model"] != "poke_net") else 0
    if freeze_epochs:
        set_backbone_trainable(model, False)
        print(f"[train] phase 1: backbone frozen for {freeze_epochs} epoch(s)")
    optimizer = build_optimizer(cfg, model)

    # ---- 5. the learning-rate schedule, sized to the WHOLE run up front ------
    # cosine_warmup needs to know the total number of optimizer steps in
    # advance to shape its decay curve correctly - that total is batches-per-
    # epoch times the number of epochs, and warmup lasts warmup_ratio (default
    # 5%) of it.
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * cfg["epochs"]
    scheduler = cosine_warmup(optimizer, int(total_steps * cfg["warmup_ratio"]), total_steps)

    history = HistoryLogger(out_dir / "history.csv")
    stopper = EarlyStopping(patience=cfg["patience"])
    best_top1, start_epoch = 0.0, 1

    # ---- 6. optionally resume from a checkpoint -------------------------------
    if cfg["resume"] and Path(cfg["resume"]).exists():
        ck = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state"):
            optimizer.load_state_dict(ck["optimizer_state"])
        # Continue counting epochs from right after the checkpoint - not the
        # optimizer/scheduler state, which was already freshly rebuilt above,
        # so a resume is a genuine "warm restart" (fresh LR schedule over the
        # weights as they were left) rather than a byte-for-byte continuation.
        # See the README's "Resuming / extending a run" section for the
        # reasoning and the early-stopping trap this can spring.
        start_epoch = ck.get("epoch", 0) + 1
        best_top1 = ck.get("metrics", {}).get("top1", 0.0)
        print(f"[train] resumed from {cfg['resume']} at epoch {start_epoch}")

    # Snapshot the exact settings this run used, next to its checkpoints - so
    # months later, `outputs/<run>/config.json` answers "what config produced
    # this model" without having to remember or guess.
    (out_dir / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}, indent=2),
        encoding="utf-8",
    )

    # ---- 7. the training loop itself ------------------------------------------
    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # ---- transfer-learning phase 2: unfreeze, at the configured epoch ----
        if freeze_epochs and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, True)
            # A fresh optimizer is needed because unfreezing changes WHICH
            # parameters require gradients - param_groups (called inside
            # build_optimizer) needs to re-scan the model to pick up the
            # newly-unfrozen backbone parameters into their own low-LR group.
            optimizer = build_optimizer(cfg, model)
            # A short new warmup (2%) for phase 2's own LR schedule, sized to
            # however many epochs are left - this is the SAME cosine_warmup
            # mechanism used for the whole run, just re-applied to a shorter
            # remaining window now that the backbone has joined training.
            remaining = steps_per_epoch * (cfg["epochs"] - epoch + 1)
            scheduler = cosine_warmup(optimizer, int(remaining * 0.02), remaining)
            print(f"[train] phase 2: backbone unfrozen (lr_backbone={cfg['lr_backbone']:.1e})")

        # One full pass over the training set, then one full pass over the
        # validation set - the two functions doing the actual math live in
        # src/engine.py; everything above here was just setup.
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

        # last.pt: unconditionally overwritten every epoch - always the most
        # recent state, useful for resuming a run that gets interrupted.
        # best.pt: only overwritten when THIS epoch's val top-1 beats every
        # epoch seen so far (across the whole history, resumes included,
        # since best_top1 is carried forward from the checkpoint on resume) -
        # this is the file every other script (evaluate/predict/export_onnx)
        # defaults to, precisely because it can never silently regress.
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
