"""Step 6 - training/validation loops, metrics, checkpointing.

Everything model-agnostic and dataset-agnostic lives here: the actual "run one
epoch of training", "measure accuracy on a held-out set", "save/load a
checkpoint" logic. src/train.py is the conductor that calls these functions in
the right order with the right config; this file doesn't know or care what
config produced its inputs.
"""
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
    """A running mean, updated incrementally instead of storing every value.

    `loss.item()` for every batch could be collected into a list and averaged
    at the end, but that means holding thousands of floats in memory for no
    reason - this just keeps a running sum and count instead, which is exactly
    equivalent and effectively free.
    """

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # n is the batch size - a batch's average loss needs to be weighted
        # by how many images it represents, or a half-empty final batch would
        # be counted as equally important as a full one.
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


@torch.no_grad()
def topk_correct(logits: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    """Count, per batch, how many predictions were correct within the top k.

    Top-1: was the single highest-scoring class the right one?
    Top-5: was the right class ANYWHERE in the 5 highest-scoring predictions?

    Top-5 matters here because some Pokemon are genuinely hard to tell apart
    (Nidorina vs Nidorino, adjacent evolution stages) - a model that puts the
    correct answer in 2nd place, just behind a very similar-looking Pokemon,
    is doing something very different from one that has no idea at all, and
    top-1 accuracy alone can't tell those two situations apart.
    """
    maxk = min(max(ks), logits.size(1))
    # topk gives the maxk highest-scoring class indices per row, sorted best
    # first. Comparing every one of them against the true label (via
    # broadcasting: target.view(-1, 1)) produces a boolean grid; slicing that
    # grid down to just the first k columns and checking "was there a hit
    # anywhere in this row" gives exactly the top-k correctness per image.
    _, pred = logits.topk(maxk, dim=1)
    hits = pred.eq(target.view(-1, 1))
    return {k: int(hits[:, : min(k, maxk)].any(dim=1).sum().item()) for k in ks}


def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Convex-combine pairs of images and their labels.

    Strong regulariser for small datasets: it stops the network memorising
    individual sprites by never showing it a pure example twice.

    Concretely: shuffle the batch to get a second, randomly-paired image for
    each one, then blend every image with its random partner by a mix factor
    `lam` drawn fresh each batch from a Beta(alpha, alpha) distribution (with
    alpha=0.2, this usually lands close to 0 or 1 - mostly one image with a
    faint trace of another - occasionally near 0.5, a genuine blend). The loss
    then targets BOTH labels, weighted by that same lam, in train_one_epoch
    below. The model never sees a "clean" training image twn twice in exactly
    the same form, which is exactly what makes this such an effective defence
    against overfitting a few-hundred-images-per-class dataset.
    """
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)  # a random pairing, redrawn every batch
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def cosine_warmup(optimizer, warmup_steps: int, total_steps: int, min_factor: float = 0.01):
    """Build a learning-rate schedule: linear ramp-up, then a cosine decay down to (almost) zero.

    Returns a PyTorch LambdaLR, which multiplies the optimizer's base LR by
    whatever `fn(step)` returns at each step - so `fn` below IS the shape of
    the whole learning-rate curve, expressed as a multiplier:

      * step < warmup_steps: ramps linearly from ~0 up to 1.0. Starting a
        model - especially a freshly-initialised head - at full learning rate
        immediately risks a huge, destabilising first update; a short warmup
        lets the optimizer's internal statistics (particularly Adam's moment
        estimates) settle in gently first.
      * step >= warmup_steps: follows a cosine curve down from 1.0 to
        min_factor (never all the way to exactly zero) over the remaining
        steps. Unlike a step-wise LR drop, the cosine's decay is smooth and
        naturally spends more time at both the "still learning fast" end and
        the "fine-tuning gently" end, spending less time in an awkward
        middle ground.

    Called once per epoch's worth of *batches*, not once per epoch: `.step()`
    is invoked inside train_one_epoch's per-batch loop below, which is what
    makes this a smoothly changing curve rather than a staircase that only
    moves once per epoch.
    """
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return LambdaLR(optimizer, fn)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None,
                    scheduler=None, mixup_alpha: float = 0.0, clip_grad: float = 0.0,
                    desc: str = "train") -> dict[str, float]:
    """One full pass over the training set: forward, loss, backward, update - per batch.

    Every batch runs the same six-step cycle any PyTorch training loop does:
      1. zero_grad   - clear gradients left over from the previous batch
      2. forward     - run the images through the model to get predictions ("logits")
      3. loss        - compare predictions against the true labels
      4. backward    - compute how much each weight contributed to that loss
      5. step        - nudge every weight a little in the direction that reduces loss
      6. (scheduler) - update the learning rate for the next batch
    """
    model.train()  # enables dropout and lets BatchNorm use this batch's own statistics
    loss_m, seen, correct = AverageMeter(), 0, {1: 0, 5: 0}
    amp = scaler is not None and device.type == "cuda"

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        # non_blocking=True lets the CPU->GPU copy overlap with other work
        # when the data came from pinned memory (see dataset.make_loaders).
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # autocast runs the forward pass in float16 where safe (much faster,
        # less GPU memory) while keeping numerically-sensitive parts in
        # float32 automatically - this is what "amp" / mixed precision means.
        # It only activates when there's a GradScaler AND a CUDA device;
        # CPU/MPS training always runs in plain float32.
        with torch.autocast(device_type=device.type, enabled=amp):
            if mixup_alpha > 0:
                images, ya, yb, lam = mixup_batch(images, targets, mixup_alpha)
                logits = model(images)
                # Since each mixed image is lam-parts-image-A and
                # (1-lam)-parts-image-B, the loss is blended the same way:
                # partial credit against both original labels.
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits = model(images)
                loss = criterion(logits, targets)

        if amp:
            # float16 gradients can underflow to zero for very small values.
            # GradScaler multiplies the loss up before backward() (so small
            # gradients survive in float16) and un-scales it again before the
            # optimizer step - standard mixed-precision training practice.
            scaler.scale(loss).backward()
            if clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad:
                # Caps the overall size ("norm") of the gradient update.
                # Without this, an unusually hard batch (or the LR spike right
                # after a warm restart - see the README's resuming section)
                # can occasionally produce a gradient large enough to knock
                # the model badly off course in one step; clipping is a cheap
                # insurance policy against that.
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        if scheduler is not None:
            # Stepped every BATCH, not every epoch - see cosine_warmup's
            # docstring for why that matters for the shape of the LR curve.
            scheduler.step()

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        hits = topk_correct(logits.detach().float(), targets)
        correct[1] += hits[1]
        correct[5] += hits[5]
        seen += bs
        # Live progress readout - updated every batch so you can watch loss,
        # accuracy and the current LR move in real time during a long epoch.
        pbar.set_postfix(loss=f"{loss_m.avg:.3f}", top1=f"{correct[1] / seen:.3f}",
                         lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return {"loss": loss_m.avg, "top1": correct[1] / max(seen, 1), "top5": correct[5] / max(seen, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "val", return_preds: bool = False):
    """Measure the model on a held-out set - no learning happens here.

    Structurally similar to train_one_epoch's loop, but with three important
    differences that all follow from "this is measurement, not learning":
      * model.eval() - dropout is switched off and BatchNorm uses its learned
        running statistics instead of this batch's own, so the SAME input
        always produces the SAME output (a training-mode forward pass would
        not be reproducible, because dropout is random).
      * @torch.no_grad() - gradients are never needed for evaluation, so
        skipping their computation entirely saves both memory and time.
      * no optimizer, no backward(), no scheduler - nothing here changes a
        single model weight.
    """
    model.eval()
    loss_m, seen, correct = AverageMeter(), 0, {1: 0, 5: 0}
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        logits = logits.float()  # back to float32 before any further math, for numerical stability

        bs = targets.size(0)
        loss_m.update(loss.item(), bs)
        hits = topk_correct(logits, targets)
        correct[1] += hits[1]
        correct[5] += hits[5]
        seen += bs
        if return_preds:
            # Only collected when a caller actually needs per-image detail
            # (src/evaluate.py, for the confusion matrix and mistake grid) -
            # skipped during ordinary training-time validation, where only
            # the aggregate loss/accuracy numbers above are needed.
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
    """Stop when val loss has not improved for `patience` epochs.

    A resumed run creates a NEW EarlyStopping instance (see src/train.py),
    which starts fresh with no memory of the run before it - its very first
    reading becomes the target every later epoch has to beat. That matters a
    lot for a "warm restart" resume, where the learning rate deliberately
    jumps back up and val loss gets temporarily WORSE before it (hopefully)
    recovers - patience needs to be generous enough to survive that dip. See
    the README's "Resuming / extending a run" section for a worked example of
    this exact trap.
    """
    patience: int = 10
    min_delta: float = 1e-4  # how much better counts as "actually improved", not just noise
    best: float = field(default=float("inf"))
    counter: int = 0

    def step(self, value: float) -> bool:
        """Call once per epoch with the latest val loss; returns True when it's time to stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class HistoryLogger:
    """Per-epoch metrics, written to CSV as training progresses.

    Loads any existing file at construction so resuming a run (--resume) appends
    to prior epochs instead of silently overwriting them - without this, epoch 61
    of a resumed run would wipe out the logged history for epochs 1-60.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if self.path.exists():
            # Every value round-trips through CSV as text, so it has to be
            # cast back: `epoch` to int (used for sorting/dedup below),
            # everything else to float (loss/accuracy/lr/elapsed_s).
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self.rows.append({
                        k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()
                    })

    def log(self, row: dict) -> None:
        # A resumed run repeats its start epoch if interrupted mid-epoch; keep
        # only the latest record for any given epoch number rather than duplicating it.
        self.rows = [r for r in self.rows if r["epoch"] != row["epoch"]]
        self.rows.append(row)
        self.rows.sort(key=lambda r: r["epoch"])
        # Rewrites the WHOLE file every call rather than appending a line.
        # Simple and safe (a crash mid-write never leaves a half-appended,
        # unparseable CSV row) - and at one row per epoch, cheap enough not
        # to matter even over a few hundred epochs.
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


def save_checkpoint(path: Path, model, optimizer=None, epoch: int = 0,
                    metrics: dict | None = None, config: dict | None = None,
                    class_names: list[str] | None = None) -> None:
    """Write everything needed to either resume training or run inference later, in one file.

    A checkpoint bundles more than just the model's weights:
      * model_state      - the actual learned weights (what most people mean by "the model")
      * optimizer_state  - Adam/SGD's internal momentum, needed to resume
                           training smoothly rather than restarting the optimizer cold
      * epoch             - so a --resume run knows where to continue counting from
      * metrics           - this epoch's val loss/accuracy, so --resume knows
                           what "best so far" to try to beat
      * config            - the full training config, so load_model_from_checkpoint
                           can rebuild the exact right architecture before loading weights into it
      * class_names       - so a prediction (a bare list of 151 numbers) can be
                           turned back into an actual Pokemon name, without
                           needing classes.json to also be present
    This is what makes a single .pt file self-describing: src/predict.py,
    src/evaluate.py and src/export_onnx.py can all load one with no other
    context and know exactly what model produced it and how to rebuild it.
    """
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
    """Load a checkpoint dict back off disk.

    `weights_only=False` is required here because a checkpoint holds more than
    tensors (config is a plain dict, class_names a list of strings) - PyTorch's
    stricter weights_only=True mode is meant for loading model weights alone
    from an untrusted source and would reject this richer, but only
    self-produced, format.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
