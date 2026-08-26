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
from src.models import load_model_from_checkpoint
from src.utils import IMAGE_EXTS, get_device, load_image_rgb


def download(url: str, dest_dir: Path | None = None) -> Path:
    """Grab an image URL to a local file. Most hosts 403 the default Python UA."""
    dest_dir = dest_dir or Path(tempfile.gettempdir())
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download"
    if not any(name.lower().endswith(e) for e in IMAGE_EXTS):
        name += ".jpg"
    dest = dest_dir / name
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PokeNet)"})
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



# --- result window -----------------------------------------------------------
# Single hue, light -> dark: the bars encode magnitude, so the winner is simply
# the darkest step of the same blue rather than a second colour.
INK, MUTED, GRID = "#1a1d24", "#6b7280", "#e4e8ee"
BAR, BAR_TOP, SURFACE = "#b9cbdb", "#2c6aa8", "#ffffff"


def show_predictions(results, names, topk: int = 5, max_windows: int = 12) -> None:
    """Open a window per prediction: the image, the answer, and the top-k bars."""
    import matplotlib.pyplot as plt  # imported late so --no-show stays headless
    from matplotlib.patches import FancyBboxPatch

    if not plt.get_backend() or plt.get_backend().lower() == "agg":
        print("[predict] no interactive display available - skipping windows")
        return

    shown = results[:max_windows]
    if len(results) > max_windows:
        print(f"[predict] showing the first {max_windows} of {len(results)} results")

    for path, idxs, confs in shown:
        k = min(topk, len(idxs))
        labels = [names[i] for i in idxs[:k]][::-1]
        values = [c for c in confs[:k]][::-1]

        fig, (ax_img, ax_bar) = plt.subplots(
            1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
        fig.patch.set_facecolor(SURFACE)

        ax_img.imshow(load_image_rgb(path))
        ax_img.axis("off")

        # The answer, stated plainly.
        fig.text(0.5, 0.955, names[idxs[0]], ha="center", va="top",
                 fontsize=24, fontweight="bold", color=INK)
        fig.text(0.5, 0.885, f"{confs[0] * 100:.1f}% confident   ·   {Path(path).name}",
                 ha="center", va="top", fontsize=10, color=MUTED)

        ypos = range(k)
        ax_bar.barh(list(ypos), values, height=0.62,
                    color=[BAR_TOP if i == k - 1 else BAR for i in range(k)])
        for y, (v, lab) in enumerate(zip(values, labels)):
            ax_bar.text(v + 0.015, y, f"{v * 100:.1f}%", va="center",
                        fontsize=9, color=MUTED)
            ax_bar.text(-0.015, y, lab, va="center", ha="right",
                        fontsize=10, color=INK)
        ax_bar.set_xlim(0, 1.16)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        ax_bar.set_facecolor(SURFACE)
        for side in ("top", "right", "bottom", "left"):
            ax_bar.spines[side].set_visible(False)
        ax_bar.set_title(f"top {k}", loc="left", fontsize=9, color=MUTED, pad=8)

        fig.subplots_adjust(left=0.02, right=0.97, top=0.80, bottom=0.06, wspace=0.45)

    print(f"[predict] close the window(s) to continue")
    plt.show()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--url", action="append", help="image URL; repeatable")
    ap.add_argument("--save-to", type=Path, help="where --url downloads land (default: temp)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tta", action="store_true", help="average logits with the mirrored image")
    ap.add_argument("--no-show", dest="show", action="store_false",
                    help="do not open result windows (default: open them)")
    ap.set_defaults(show=True)
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

    if args.show:
        show_predictions(results, names, args.topk)

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
