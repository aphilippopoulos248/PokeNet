"""Step 9 - inference on new images.

    python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --image path/to/pic.jpg
    python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --dir my_photos --topk 3 --csv out.csv
    python -m src.predict --checkpoint outputs/resnet18_ft/best.pt --url https://example.com/pikachu.png
"""
from __future__ import annotations

import argparse
import csv
import tempfile
import urllib.request
from pathlib import Path

import torch

from src.dataset import build_transforms
from src.evaluate import load_model_from_checkpoint
from src.utils import IMAGE_EXTS, get_device, load_image_rgb


def download(url: str, dest_dir: Path | None = None) -> Path:
    """Grab an image URL to a local file. Most hosts 403 the default Python UA."""
    dest_dir = dest_dir or Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download"
    if not any(name.lower().endswith(e) for e in IMAGE_EXTS):
        name += ".jpg"
    dest = dest_dir / name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PokemonCNN)"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"[predict] downloaded {url} -> {dest}")
    return dest


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
    ap.add_argument("--url", action="append", help="image URL; repeatable")
    ap.add_argument("--save-to", type=Path, help="where --url downloads land (default: temp)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tta", action="store_true", help="average logits with the mirrored image")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not (args.image or args.dir or args.url):
        ap.error("pass --image, --dir or --url")

    device = get_device(args.device)
    model, names, cfg = load_model_from_checkpoint(args.checkpoint, device)
    tf = build_transforms(int(cfg.get("img_size", 224)), train=False)

    paths: list[Path] = []
    if args.image:
        paths.append(args.image)
    if args.url:
        paths += [download(u, args.save_to) for u in args.url]
    if args.dir:
        paths += sorted(p for p in args.dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
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
