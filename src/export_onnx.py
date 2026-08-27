"""Step 11 - export a checkpoint to ONNX for serverless hosting.

PyTorch is ~800 MB installed and takes seconds to import; ONNX Runtime is ~15 MB
and starts in about one. On a serverless host that is the difference between a
deploy that fits and one that does not.

    python -m src.export_onnx --checkpoint outputs/resnet18_ft/best.pt

Writes deploy/vercel/model/{pokemon.onnx, meta.json} and copies web/index.html
into the deploy folder, then verifies ONNX and PyTorch agree.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from src.models import load_model_from_checkpoint
from src.types import dex_map, type_map
from src.utils import IMAGENET_MEAN, IMAGENET_STD, ROOT, rel_to_root

DEPLOY = ROOT / "deploy" / "vercel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=DEPLOY / "model")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.exists():
        print(f"[onnx] checkpoint not found: {args.checkpoint}")
        return 1

    model, names, cfg = load_model_from_checkpoint(args.checkpoint, "cpu")
    img_size = int(cfg.get("img_size", 224))
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "pokemon.onnx"

    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    meta = {
        "class_names": names,
        "img_size": img_size,
        "resize_to": int(img_size * 1.14),   # must match build_transforms(train=False)
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "model": cfg.get("model", "unknown"),
        "checkpoint": rel_to_root(args.checkpoint),
        # Baked in so the serverless function stays self-contained: it has no
        # src/ and no metadata CSV, only this folder.
        "types": type_map(names),
        "dex": dex_map(names),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    src_page = ROOT / "web" / "index.html"
    if src_page.exists():
        shutil.copy2(src_page, DEPLOY / "index.html")

    mb = onnx_path.stat().st_size / 1024**2
    print(f"[onnx] wrote {onnx_path} ({mb:.1f} MB)")
    typed = sum(1 for v in meta["types"].values() if v)
    print(f"[onnx] wrote {args.out / 'meta.json'} ({len(names)} classes, {img_size}px, "
          f"{typed}/{len(names)} with types)")
    print(f"[onnx] copied web/index.html -> {DEPLOY / 'index.html'}")

    if not args.skip_verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[onnx] onnxruntime not installed - skipping parity check "
                  "(pip install onnxruntime)")
            return 0
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        x = torch.randn(2, 3, img_size, img_size)
        with torch.no_grad():
            torch_out = model(x).numpy()
        onnx_out = sess.run(None, {"image": x.numpy()})[0]
        diff = float(np.abs(torch_out - onnx_out).max())
        agree = (torch_out.argmax(1) == onnx_out.argmax(1)).all()
        print(f"[onnx] parity: max abs diff {diff:.2e}, same argmax: {bool(agree)}")
        if diff > 1e-3 or not agree:
            print("[onnx] WARNING - outputs diverge, do not deploy this file")
            return 1

    print(f"\n[onnx] next: cd deploy/vercel && vercel deploy --prod")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
