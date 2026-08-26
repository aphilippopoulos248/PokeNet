"""Step 8 - evaluate on the held-out test set.

    python -m src.evaluate --checkpoint outputs/resnet18_ft/best.pt

Produces top-1/top-5, a per-class report, a confusion matrix, the most-confused
class pairs, and a grid of the model's worst mistakes. The confused pairs are
where the real insight is - Nidoran-M vs Nidoran-F, the Eeveelutions, and the
three-stage starters are the classic failure modes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import PokemonDataset, build_transforms, denormalize
from src.engine import evaluate as run_eval
from src.models import load_model_from_checkpoint
from src.utils import DATA_SPLITS, REPORTS, describe_device, get_device


def confusion_figure(cm: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(np.log1p(cm), cmap="magma")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def mistakes_figure(ds, preds, targets, probs, names, out: Path, n: int = 24) -> None:
    wrong = np.where(preds != targets)[0]
    if len(wrong) == 0:
        return
    conf = probs[wrong, preds[wrong]]
    worst = wrong[np.argsort(-conf)][:n]  # confidently wrong = most informative
    cols = 6
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.8))
    for ax, i in zip(np.array(axes).ravel(), worst):
        img, _ = ds[int(i)]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"T:{names[targets[i]][:11]}\nP:{names[preds[i]][:11]} {probs[i, preds[i]]:.2f}",
                     fontsize=7, color="#b3261e")
        ax.axis("off")
    for ax in np.array(axes).ravel()[len(worst):]:
        ax.axis("off")
    fig.suptitle("Confidently wrong predictions", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--splits-dir", type=Path, default=DATA_SPLITS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = get_device(args.device)
    model, names, cfg = load_model_from_checkpoint(args.checkpoint, device)
    img_size = int(cfg.get("img_size", 224))

    ds = PokemonDataset(args.splits_dir / f"{args.split}.csv", build_transforms(img_size, False))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=device.type == "cuda")

    print(f"[eval] {describe_device(device)} | {cfg.get('model')} | {len(ds)} {args.split} images")
    metrics, (preds, targets, probs) = run_eval(model, dl, nn.CrossEntropyLoss(), device,
                                                desc=args.split, return_preds=True)
    print(f"\n[eval] loss {metrics['loss']:.4f} | top-1 {metrics['top1']:.4f} | top-5 {metrics['top5']:.4f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"{args.checkpoint.parent.name}_{args.split}"

    report = classification_report(targets, preds, labels=range(len(names)),
                                   target_names=names, zero_division=0, digits=3)
    (REPORTS / f"{stem}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(targets, preds, labels=range(len(names)))
    confusion_figure(cm, REPORTS / f"{stem}_confusion.png",
                     f"{cfg.get('model')} - {args.split} confusion (top-1 {metrics['top1']:.3f})")

    per_class = cm.diagonal() / np.maximum(cm.sum(1), 1)
    order = np.argsort(per_class)
    print("\nWeakest 15 classes:")
    for i in order[:15]:
        print(f"  {names[i]:<16} acc {per_class[i]:.2f}  ({int(cm[i].sum())} test images)")

    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:15]
    print("\nMost-confused pairs (true -> predicted):")
    for a, b in pairs:
        if off[a, b] == 0:
            break
        print(f"  {names[a]:<16} -> {names[b]:<16} {off[a, b]}x")

    mistakes_figure(ds, preds, targets, probs, names, REPORTS / f"{stem}_mistakes.png")
    print(f"\n[eval] wrote reports to {REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
