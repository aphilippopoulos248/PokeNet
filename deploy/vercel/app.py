"""Vercel entrypoint - serves the page AND runs inference, via ONNX Runtime.

`app.py` at the project root is one of Vercel's default Python entrypoint
locations, auto-detected because `flask` is listed in requirements.txt. Once
detected, this Flask app owns every route on the domain - "/" included - so
unlike the old per-file api/ convention, the page and the API live in one
place on purpose.

Model + metadata load once per warm container (module import time), not once
per request.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MAX_BYTES = 4 * 1024 * 1024  # Vercel caps request bodies at 4.5 MB total

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES + 256 * 1024  # headroom for multipart overhead

_META = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
_SESSION = ort.InferenceSession(str(MODEL_DIR / "pokemon.onnx"), providers=["CPUExecutionProvider"])
_MEAN = np.array(_META["mean"], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(_META["std"], dtype=np.float32).reshape(3, 1, 1)
# {class_name: ["Fire", "Flying"]}, baked in by src/export_onnx.py. Absent on a
# meta.json exported before types existed, hence the .get - an older deploy
# degrades to no background colour rather than a 500.
_TYPES = _META.get("types", {})
# {class_name: national dex number} - the page turns this into a 3D model URL.
_DEX = _META.get("dex", {})


def load_rgb(data: bytes) -> Image.Image:
    """Composite alpha onto white - never .convert('RGB') a transparent sprite."""
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    """Mirrors src/dataset.build_transforms(train=False) exactly."""
    size, short = _META["img_size"], _META["resize_to"]
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - _MEAN) / _STD)[None, ...]


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/style.css")
def stylesheet():
    # This Flask app owns every route on the domain, so the stylesheet needs an
    # explicit route - there is no static file server sitting behind it.
    return send_from_directory(ROOT, "style.css")


@app.get("/api/info")
def info():
    return jsonify({
        "model": _META.get("model"),
        "checkpoint": _META.get("checkpoint"),
        "classes": len(_META["class_names"]),
        "device": "cpu (onnxruntime)",
        "img_size": _META["img_size"],
    })


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "No image was uploaded."}), 400

    data = file.read()
    if not data:
        return jsonify({"error": "That file was empty."}), 400

    try:
        x = preprocess(load_rgb(data))
    except Exception:  # noqa: BLE001
        return jsonify({"error": "That file isn't an image we can read. "
                                 "Try a JPG, PNG or WEBP."}), 400

    t0 = time.perf_counter()
    logits = _SESSION.run(None, {"image": x})[0][0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    names = _META["class_names"]
    k = min(5, len(names))
    order = np.argsort(-probs)[:k]
    return jsonify({
        "predictions": [
            {
                "name": names[i],
                "confidence": float(probs[i]),
                "types": _TYPES.get(names[i], []),
                "dex": _DEX.get(names[i]),
            }
            for i in order
        ],
        "elapsed_ms": round(elapsed_ms, 1),
    })


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": "That image is larger than 4 MB."}), 413
