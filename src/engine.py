"""Step 6 - training/validation loops, metrics, checkpointing."""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    maxk = min(max(ks), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    hits = pred.eq(target.view(-1, 1))
    return {k: int(hits[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks}


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Convex-combine pairs of images and their labels.

    Strong regulariser for small datasets: it stops the network memorising
    individual sprites by never showing it a pure example twice.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_factor: float = 0.01):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    scheduler=None, mixup_alpha: float = 0.0, clip_grad: float = 0.0,
                    desc: str = "train") -> dict[str, float]:
    model.train()
    loss_m, seen, correct = AverageMeter(), 0, {1: 0, 5: 0}
    amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            if mixup_alpha > 0:
                images, ya, yb, lam = mixup_batch(images, targets, mixup_alpha)
                logits = model(images)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        if amp:
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        hits = topk_correct(logits.detach().float(), targets)
        correct[1] += hits[1]
        correct[5] += hits[5]
        seen += bs
        pbar.set_postfix(loss=f"{loss_m.avg:.3f}", top1=f"{correct[1] / seen:.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return {"loss": loss_m.avg, "top1": correct[1] / max(seen, 1), "top5": correct[5] / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "val", return_preds: bool = False):
    model.eval()
    loss_m, seen, correct = AverageMeter(), 0, {1: 0, 5: 0}
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        logits = logits.float()

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        hits = topk_correct(logits, targets)
        correct[1] += hits[1]
        correct[5] += hits[5]
        seen += bs
        if return_preds:
            all_preds.append(logits.argmax(1).cpu())
            all_targets.append(targets.cpu())
            all_probs.append(logits.softmax(1).cpu())

    metrics = {"loss": loss_m.avg, "top1": correct[1] / max(seen, 1), "top5": correct[5] / max(seen, 1)}
    if return_preds:
        return metrics, (torch.cat(all_preds).numpy(), torch.cat(all_targets).numpy(),
                         torch.cat(all_probs).numpy())
    return metrics


@dataclass
class EarlyStopping:
    """Stop when val loss has not improved for `patience` epochs."""
    patience: int = 10
    min_delta: float = 1e-4
    best: float = field(default=float("inf"))
    counter: int = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class HistoryLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def log(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def save_checkpoint(path: Path, model, optimizer=None, epoch: int = 0,
                    metrics: dict | None = None, config: dict | None = None,
                    class_names: list[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "class_names": class_names or [],
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
