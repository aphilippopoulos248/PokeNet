"""Step 10 - a local web app so other people can use the model.

An HTML page cannot run PyTorch, so this serves the page AND does the inference:
the browser posts an image, this process runs it through the checkpoint on your
GPU and returns JSON.

    pip install flask
    python -m src.app
    # then open http://127.0.0.1:5000

    python -m src.app --checkpoint outputs/resnet18_ft/best.pt --port 8080 --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory

from src.dataset import build_transforms
from src.models import load_model_from_checkpoint
from src.types import types_for
from src.utils import ROOT, describe_device, get_device, load_image_rgb, rel_to_root

WEB_DIR = ROOT / "web"
MAX_UPLOAD_MB = 16

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

STATE: dict = {}


def default_checkpoint() -> Path | None:
    """Prefer a fine-tuned model, fall back to whatever exists."""
    preferred = [
        ROOT / "outputs" / "resnet50_ft" / "best.pt",
        ROOT / "outputs" / "resnet18_ft" / "best.pt",
        ROOT / "outputs" / "baseline_pokenet" / "best.pt",
    ]
    for p in preferred:
        if p.exists():
            return p
    found = sorted((ROOT / "outputs").glob("*/best.pt"))
    return found[0] if found else None


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/api/info")
def info():
    return jsonify({
        "model": STATE["cfg"].get("model", "unknown"),
        "checkpoint": rel_to_root(STATE["checkpoint"]),
        "classes": len(STATE["names"]),
        "device": describe_device(STATE["device"]),
        "img_size": STATE["img_size"],
        "val_top1": STATE["cfg"].get("_val_top1"),
    })


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "No image was uploaded."}), 400

    raw = file.read()
    if not raw:
        return jsonify({"error": "That file was empty."}), 400

    try:
        img = load_image_rgb(io.BytesIO(raw))
    except Exception:  # noqa: BLE001
        return jsonify({"error": "That file isn't an image we can read. "
                                 "Try a JPG, PNG or WEBP."}), 400

    device, model = STATE["device"], STATE["model"]
    tensor = STATE["tf"](img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(tensor).float()
            if STATE["tta"]:
                logits = (logits + model(torch.flip(tensor, dims=[3])).float()) / 2
        probs = logits.softmax(1)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    k = min(5, probs.numel())
    conf, idx = probs.topk(k)
    # "types" is a lookup keyed off the predicted name, not a second model head -
    # see src/types.py for why that distinction matters. The page uses it to
    # colour its background.
    return jsonify({
        "predictions": [
            {
                "name": STATE["names"][i],
                "confidence": float(c),
                "types": types_for(STATE["names"][i]),
            }
            for i, c in zip(idx.tolist(), conf.tolist())
        ],
        "elapsed_ms": round(elapsed_ms, 1),
    })


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": f"That image is larger than {MAX_UPLOAD_MB} MB."}), 413


def main() -> int:
    ap = argparse.ArgumentParser(description="Serve the Pokedex web app.")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 exposes it to your local network")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.set_defaults(tta=True)
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve() if args.checkpoint else default_checkpoint()
    if ckpt is None or not Path(ckpt).exists():
        print("[app] no checkpoint found in outputs/ - train a model first:\n"
              "      python -m src.train --config configs/resnet18.yaml")
        return 1

    device = get_device(args.device)
    model, names, cfg = load_model_from_checkpoint(ckpt, device)
    img_size = int(cfg.get("img_size", 224))

    STATE.update(model=model, names=names, cfg=cfg, device=device, checkpoint=Path(ckpt),
                 img_size=img_size, tf=build_transforms(img_size, train=False), tta=args.tta)

    print(f"\n[app] {cfg.get('model')} · {len(names)} classes · {describe_device(device)}")
    print(f"[app] checkpoint {rel_to_root(ckpt)}")
    print(f"[app] open  http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    if args.host == "0.0.0.0":
        print("[app] reachable from your local network - anyone on your wifi can use it")
    print("[app] ctrl-c to stop\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
