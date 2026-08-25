"""Step 3 - build train/val/test manifests.

Splitting is done ONCE here and frozen to CSV, so every experiment sees the same
test set. Two things happen before the split:
  * class folder names are folded onto the canonical 151 (see src.names), so a
    merge of two datasets does not split Nidoran-f and Nidoran(female) in two
  * exact duplicates are dropped - a duplicate straddling train and test leaks
    the answer and inflates your accuracy

    python -m src.prepare --val 0.15 --test 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.inspect_data import find_class_dirs
from src.names import resolve
from src.utils import DATA_RAW, DATA_SPLITS, ROOT, write_json


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def stratified_split(df: pd.DataFrame, val: float, test: float, seed: int) -> pd.DataFrame:
    """Per-class shuffle-and-slice. Guarantees every class appears in train."""
    frames = []
    for _, grp in df.groupby("class", sort=True):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(grp)
        n_test = int(round(n * test))
        n_val = int(round(n * val))
        if n >= 3:  # never let val/test eat a class alive
            n_test = max(1, min(n_test, n - 2))
            n_val = max(1, min(n_val, n - 1 - n_test))
        else:
            n_test = n_val = 0
        grp["split"] = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA_RAW)
    ap.add_argument("--out", type=Path, default=DATA_SPLITS)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="keep folders that do not map to one of the official 151")
    ap.add_argument("--min-side", type=int, default=16)
    args = ap.parse_args()

    folders = find_class_dirs(args.root)
    if not folders:
        print(f"[prepare] no images under {args.root} - run `python -m src.download`")
        return 1

    mapping, unmatched = resolve(list(folders))
    if unmatched:
        print(f"[prepare] {len(unmatched)} folder(s) do not match the official 151: "
              + ", ".join(unmatched[:12]) + ("..." if len(unmatched) > 12 else ""))
        if args.keep_unmatched:
            for f in unmatched:
                mapping[f] = f
        else:
            print("[prepare] dropping them (pass --keep-unmatched to keep)")

    rows, dropped_bad, dropped_dupe, seen = [], 0, 0, {}
    for folder, files in tqdm(folders.items(), desc="validating", unit="folder"):
        cls = mapping.get(folder)
        if cls is None:
            continue
        for f in files:
            try:
                with Image.open(f) as im:
                    w, h = im.size
                if min(w, h) < args.min_side:
                    dropped_bad += 1
                    continue
            except Exception:  # noqa: BLE001
                dropped_bad += 1
                continue
            if not args.keep_duplicates:
                key = file_hash(f)
                if key in seen:
                    dropped_dupe += 1
                    continue
                seen[key] = f
            rows.append({"path": f.relative_to(ROOT).as_posix(), "class": cls})

    if not rows:
        print("[prepare] nothing usable survived - check data/raw")
        return 1

    df = pd.DataFrame(rows)
    class_names = sorted(df["class"].unique())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df["label"] = df["class"].map(class_to_idx)
    df = stratified_split(df, args.val, args.test, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        df[df["split"] == split].reset_index(drop=True).to_csv(args.out / f"{split}.csv", index=False)
    write_json({"class_names": class_names, "class_to_idx": class_to_idx,
                "num_classes": len(class_names)}, args.out / "classes.json")

    counts = df.groupby("split").size().to_dict()
    per_class = df[df["split"] == "train"].groupby("class").size()
    print(
        f"\n[prepare] {len(df)} usable images across {len(class_names)} classes\n"
        f"[prepare] dropped: {dropped_bad} unreadable/tiny, {dropped_dupe} duplicates\n"
        f"[prepare] train {counts.get('train', 0)} | val {counts.get('val', 0)} | test {counts.get('test', 0)}\n"
        f"[prepare] train per class: min {per_class.min()}, median {int(per_class.median())}, max {per_class.max()}\n"
        f"[prepare] wrote {args.out}/train.csv, val.csv, test.csv, classes.json"
    )
    if len(class_names) != 151:
        missing = 151 - len(class_names)
        print(f"[prepare] NOTE: {len(class_names)} classes, not 151 "
              f"({missing} of the original 151 have no images in this dataset)")
    print("[prepare] next: python -m src.train --config configs/resnet18.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
