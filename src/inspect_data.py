"""Step 2 - look at the data before you train on it.

Discovers the folder layout (we do not assume one), then reports class balance,
image sizes, colour modes, corrupt files and exact duplicates. Writes
reports/dataset_report.md and reports/class_distribution.png.

    python -m src.inspect_data
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from src.utils import DATA_RAW, IMAGE_EXTS, REPORTS


def find_class_dirs(root: Path) -> dict[str, list[Path]]:
    """Any directory that directly holds images is treated as a class.

    This survives both `raw/<class>/*.jpg` and `raw/dataset/train/<class>/*.jpg`.
    Same-named folders under different parents are merged (e.g. train/ and test/
    copies of Pikachu), which is what we want - we make our own splits later.
    """
    classes: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            classes[p.parent.name].append(p)
    return dict(sorted(classes.items()))


def file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA_RAW)
    ap.add_argument("--no-hash", action="store_true", help="skip duplicate detection (faster)")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[inspect] {args.root} does not exist - run `python -m src.download` first")
        return 1

    classes = find_class_dirs(args.root)
    if not classes:
        print(f"[inspect] no images found under {args.root}")
        return 1

    counts = {c: len(f) for c, f in classes.items()}
    total = sum(counts.values())

    sizes: Counter = Counter()
    modes: Counter = Counter()
    exts: Counter = Counter()
    alpha = 0
    corrupt: list[Path] = []
    hashes: dict[str, list[Path]] = defaultdict(list)

    all_files = [f for fs in classes.values() for f in fs]
    for f in tqdm(all_files, desc="scanning", unit="img"):
        exts[f.suffix.lower()] += 1
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
                if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                    alpha += 1
        except Exception:  # noqa: BLE001
            corrupt.append(f)
            continue
        if not args.no_hash:
            hashes[file_hash(f)].append(f)

    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    dupe_extra = sum(len(ps) - 1 for ps in dupes.values())

    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    smallest, largest = ordered[:10], ordered[-10:]
    widths = [w for (w, _), n in sizes.items() for _ in range(n)]
    heights = [h for (_, h), n in sizes.items() for _ in range(n)]

    REPORTS.mkdir(parents=True, exist_ok=True)

    # Class-balance chart - the single most useful picture before training.
    fig, ax = plt.subplots(figsize=(14, 5))
    vals = [n for _, n in sorted(counts.items())]
    ax.bar(range(len(vals)), vals, color="#3b6ea5")
    ax.axhline(total / len(counts), color="#c94f3d", ls="--", lw=1,
               label=f"mean {total / len(counts):.0f}")
    ax.set_xlabel("class (alphabetical)")
    ax.set_ylabel("images")
    ax.set_title(f"Class distribution - {len(counts)} classes, {total} images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "class_distribution.png", dpi=130)
    plt.close(fig)

    lines = [
        "# Dataset report",
        "",
        f"- Root: `{args.root}`",
        f"- Classes: **{len(counts)}**" + ("" if len(counts) == 151 else "  <- expected 151, check the layout"),
        f"- Images: **{total}**",
        f"- Mean / min / max per class: {total / len(counts):.1f} / {min(counts.values())} / {max(counts.values())}",
        f"- Imbalance ratio (max/min): {max(counts.values()) / max(min(counts.values()), 1):.1f}x",
        f"- Corrupt / unreadable: {len(corrupt)}",
        f"- Exact duplicates (extra copies): {dupe_extra}" + (" (skipped)" if args.no_hash else ""),
        f"- Images with an alpha channel: {alpha}"
        + ("  <- composite onto a background, do not .convert('RGB') naively" if alpha else ""),
        "",
        "## Image geometry",
        f"- Distinct resolutions: {len(sizes)}",
        f"- Width  min/median/max: {min(widths)} / {sorted(widths)[len(widths) // 2]} / {max(widths)}",
        f"- Height min/median/max: {min(heights)} / {sorted(heights)[len(heights) // 2]} / {max(heights)}",
        "- Most common sizes: " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common(5)),
        "- Colour modes: " + ", ".join(f"{m} ({n})" for m, n in modes.most_common()),
        "- Extensions: " + ", ".join(f"{e} ({n})" for e, n in exts.most_common()),
        "",
        "## Thinnest classes (watch these in the confusion matrix)",
        "",
        "| class | images |",
        "| --- | --- |",
        *[f"| {c} | {n} |" for c, n in smallest],
        "",
        "## Fattest classes",
        "",
        "| class | images |",
        "| --- | --- |",
        *[f"| {c} | {n} |" for c, n in reversed(largest)],
    ]
    if corrupt:
        lines += ["", "## Corrupt files", ""] + [f"- `{p}`" for p in corrupt[:50]]
    if dupes:
        lines += ["", "## Duplicate groups (first 20)", ""]
        for ps in list(dupes.values())[:20]:
            lines.append("- " + " | ".join(p.name for p in ps))

    out = REPORTS / "dataset_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:14]))
    print(f"\n[inspect] wrote {out}")
    print(f"[inspect] wrote {REPORTS / 'class_distribution.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
