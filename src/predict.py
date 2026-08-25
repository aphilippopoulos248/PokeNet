"""Step 9 - inference on new images.

    python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --image path/to/pic.jpg
    python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --dir my_photos --topk 3 --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.dataset import build_transforms
from src.evaluate import load_model_from_checkpoint
from src.utils import IMAGE_EXTS, get_device, load_image_rgb


@torch.no_grad()
def predict_paths(model, paths: list[Path], tf, device, topk: int = 5, tta: bool = False,
                  batch_size: int = 32):
    results = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i: i + batch_size]
        batch = torch.stack([tf(load_image_rgb(p)) for p in chunk]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(batch).float()
            if tta:  # average with the horizontal mirror - a free ~0.5%
                logits = (logits + model(torch.flip(batch, dims=[3])).float()) / 2
        probs = logits.softmax(1)
        conf, idx = probs.topk(min(topk, probs.size(1)), dim=1)
        for p, c, j in zip(chunk, conf.cpu(), idx.cpu()):
            results.append((p, j.tolist(), c.tolist()))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tta", action="store_true", help="average logits with the mirrored image")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not args.image and not args.dir:
        ap.error("pass --image or --dir")

    device = get_device(args.device)
    model, names, cfg = load_model_from_checkpoint(args.checkpoint, device)
    tf = build_transforms(int(cfg.get("img_size", 224)), train=False)

    paths = [args.image] if args.image else sorted(
        p for p in args.dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print("[predict] no images found")
        return 1

    results = predict_paths(model, paths, tf, device, args.topk, args.tta)

    rows = []
    for path, idxs, confs in results:
        top = " | ".join(f"{names[i]} {c:.3f}" for i, c in zip(idxs, confs))
        print(f"{path.name:<40} {top}")
        rows.append({"path": str(path), "prediction": names[idxs[0]], "confidence": round(confs[0], 4),
                     "topk": top})

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[predict] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
