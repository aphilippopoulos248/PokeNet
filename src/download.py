"""Step 1 - fetch the dataset.

NOTE: `anujckulkarni/original-151-pokemon` is a STATS TABLE (name, types, HP,
Attack...), not images. It is kept here as `--metadata` because the 151 official
names are a useful canonical label list, but you cannot train a CNN on it.

The image datasets below are the real training data. Default is `pokemon151`.
You can pass --dataset more than once; each lands in its own subfolder of
data/raw/ and stage 2/3 merge same-named class folders automatically.

    python -m src.download                                  # pokemon151
    python -m src.download --dataset pokemon151 --dataset pokemon7k # merge two
    python -m src.download --metadata                       # the stats CSV only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils import DATA_RAW, IMAGE_EXTS, ROOT

# Vetted Kaggle image datasets, folder-per-Pokemon.
IMAGE_DATASETS = {
    "pokemon151": (
        "prestigemaster/original-pokemon-151-dataset",
        "Original 151, ships its own Train/Test folders (we re-split anyway)",
    ),
    "pokemon7k": (
        "lantian773030/pokemonclassification",
        "~7,000 hand-cropped labelled images, one folder per Pokemon (~150 classes)",
    ),
    "gen1_10k": (
        "thedagger/pokemon-generation-one",
        "~10,000 Generation-I images, one folder per Pokemon",
    ),
}

METADATA_DATASET = "anujckulkarni/original-151-pokemon"
METADATA_DIR = ROOT / "data" / "metadata"


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def via_kagglehub(slug: str, dest: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[download] kagglehub not installed (pip install kagglehub)")
        return False
    try:
        print(f"[download] kagglehub -> {slug}")
        cached = Path(kagglehub.dataset_download(slug))
        print(f"[download] cached at {cached}")
        _copy_tree(cached, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[download] kagglehub failed for {slug}: {exc}")
        return False


def via_kaggle_cli(slug: str, dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    print("[download] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[download] kaggle CLI failed: {exc}")
        return False


def unzip_any(dest: Path) -> None:
    for z in list(dest.rglob("*.zip")):
        print(f"[download] unzipping {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)
        z.unlink()


def fetch(slug: str, dest: Path) -> bool:
    ok = via_kagglehub(slug, dest) or via_kaggle_cli(slug, dest)
    unzip_any(dest)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Pokemon image data.")
    ap.add_argument("--dataset", action="append", choices=sorted(IMAGE_DATASETS),
                    help="repeatable; default pokemon7k")
    ap.add_argument("--slug", help="any other Kaggle slug, e.g. user/dataset-name")
    ap.add_argument("--metadata", action="store_true",
                    help="also fetch the Gen-1 stats CSV into data/metadata/")
    ap.add_argument("--dest", type=Path, default=DATA_RAW)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true", help="show known datasets and exit")
    args = ap.parse_args()

    if args.list:
        for key, (slug, desc) in IMAGE_DATASETS.items():
            print(f"  {key:<12} {slug:<40} {desc}")
        return 0

    if args.metadata:
        fetch(METADATA_DATASET, METADATA_DIR)
        print(f"[download] metadata -> {METADATA_DIR}")

    keys = args.dataset or (["pokemon151"] if not args.slug else [])
    targets = [(k, IMAGE_DATASETS[k][0]) for k in keys]
    if args.slug:
        targets.append((args.slug.split("/")[-1], args.slug))

    for name, slug in targets:
        sub = args.dest / name
        if count_images(sub) and not args.force:
            print(f"[download] {sub} already has {count_images(sub)} images - skipping")
            continue
        fetch(slug, sub)
        print(f"[download] {name}: {count_images(sub)} images under {sub}")

    total = count_images(args.dest)
    print(f"\n[download] total images under {args.dest}: {total}")
    print("[download] note: any Train/Test folders inside a dataset are merged by\n"
          "           stage 2/3 and re-split ourselves, after de-duplication.")

    if total == 0:
        print(
            "\n[download] ERROR - no images were downloaded. A CNN needs images.\n"
            "  * Check your Kaggle token: kaggle.com -> Settings -> API -> Create New Token,\n"
            f"    saved to {Path.home() / '.kaggle' / 'kaggle.json'}\n"
            "  * See available datasets with: python -m src.download --list\n"
        )
        return 1

    print("[download] next: python -m src.inspect_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
