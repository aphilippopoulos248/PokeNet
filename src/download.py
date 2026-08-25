"""Step 1 - fetch the dataset.

Primary path: kagglehub (reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
Fallback:     the `kaggle` CLI.
If data/raw already contains images, this is a no-op unless --force is passed.

    python -m src.download
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.utils import DATA_RAW, IMAGE_EXTS

DATASET_SLUG = "anujckulkarni/original-151-pokemon"


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


def via_kagglehub(dest: Path) -> bool:
    try:
        import kagglehub
    except ImportError:
        print("[download] kagglehub not installed (pip install kagglehub)")
        return False
    try:
        print(f"[download] kagglehub -> {DATASET_SLUG}")
        cached = Path(kagglehub.dataset_download(DATASET_SLUG))
        print(f"[download] cached at {cached}")
        _copy_tree(cached, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[download] kagglehub failed: {exc}")
        return False


def via_kaggle_cli(dest: Path) -> bool:
    if shutil.which("kaggle") is None:
        print("[download] kaggle CLI not on PATH")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest), "--unzip"]
    print("[download] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[download] kaggle CLI failed: {exc}")
        return False


def unzip_any(dest: Path) -> None:
    for z in list(dest.glob("*.zip")):
        print(f"[download] unzipping {z.name}")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download the original-151-pokemon dataset.")
    ap.add_argument("--dest", type=Path, default=DATA_RAW)
    ap.add_argument("--force", action="store_true", help="re-download even if images exist")
    args = ap.parse_args()

    existing = count_images(args.dest)
    if existing and not args.force:
        print(f"[download] {existing} images already in {args.dest} - skipping (use --force to redo)")
        return 0

    ok = via_kagglehub(args.dest) or via_kaggle_cli(args.dest)
    unzip_any(args.dest)

    n = count_images(args.dest)
    if not ok and n == 0:
        print(
            "\n[download] Automatic download failed.\n"
            "  1. Create a Kaggle API token: kaggle.com -> Settings -> API -> Create New Token\n"
            f"  2. Save kaggle.json to {Path.home() / '.kaggle' / 'kaggle.json'}\n"
            "  3. Re-run, or download the zip by hand and unzip it into data/raw/\n"
        )
        return 1

    print(f"[download] done - {n} images under {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
