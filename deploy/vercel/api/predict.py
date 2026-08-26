"""Vercel Python Function: POST an image, get the top-5 back.

Runs ONNX Runtime, not PyTorch. The preprocessing here mirrors
src/dataset.build_transforms(train=False) exactly - resize the short side, centre
crop, scale to 0-1, normalise with ImageNet statistics. If those drift apart the
model silently gets worse, so the numbers come from model/meta.json rather than
being retyped.
"""
from __future__ import annotations

import io
import json
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"

# Loaded once per warm container, not per request.
_META = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
_SESSION = ort.InferenceSession(
    str(MODEL_DIR / "pokemon.onnx"),
    providers=["CPUExecutionProvider"],
)
_MEAN = np.array(_META["mean"], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(_META["std"], dtype=np.float32).reshape(3, 1, 1)
MAX_BYTES = 4 * 1024 * 1024  # Vercel caps request bodies at 4.5 MB


def load_rgb(data: bytes) -> Image.Image:
    """Composite alpha onto white - never .convert('RGB') a transparent sprite."""
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(canvas, img)
    return img.convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    size, short = _META["img_size"], _META["resize_to"]
    w, h = img.size
    scale = short / min(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)

    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))

    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((arr - _MEAN) / _STD)[None, ...]


def infer(data: bytes, topk: int = 5) -> dict:
    x = preprocess(load_rgb(data))
    t0 = time.perf_counter()
    logits = _SESSION.run(None, {"image": x})[0][0]
    elapsed = (time.perf_counter() - t0) * 1000

    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    names = _META["class_names"]
    k = min(topk, len(names))
    order = np.argsort(-probs)[:k]
    return {
        "predictions": [{"name": names[i], "confidence": float(probs[i])} for i in order],
        "elapsed_ms": round(elapsed, 1),
    }


def parse_multipart(body: bytes, content_type: str) -> bytes | None:
    """Pull the first file part out of a multipart/form-data body.

    Hand-rolled because `cgi` was removed in Python 3.13 and pulling in a parser
    for one field is not worth the bundle size.
    """
    marker = "boundary="
    if marker not in content_type:
        return None
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    sep = b"--" + boundary.encode()
    for part in body.split(sep):
        head, _, payload = part.partition(b"\r\n\r\n")
        if b"filename=" in head and payload:
            return payload.rstrip(b"\r\n").rstrip(b"--").rstrip(b"\r\n")
    return None


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires this name
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send(200, {
            "model": _META.get("model"),
            "checkpoint": _META.get("checkpoint"),
            "classes": len(_META["class_names"]),
            "device": "cpu (onnxruntime)",
            "img_size": _META["img_size"],
        })

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._send(400, {"error": "No image was uploaded."})
            if length > MAX_BYTES:
                return self._send(413, {
                    "error": "That image is too large. It should have been shrunk "
                             "before upload - try a different file."})

            body = self.rfile.read(length)
            ctype = self.headers.get("Content-Type", "")
            data = parse_multipart(body, ctype) if ctype.startswith("multipart/") else body
            if not data:
                return self._send(400, {"error": "Could not read the uploaded file."})

            try:
                result = infer(data)
            except Exception:  # noqa: BLE001
                return self._send(400, {
                    "error": "That file isn't an image we can read. Try a JPG, PNG or WEBP."})

            self._send(200, result)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"Server error: {type(exc).__name__}"})

    def log_message(self, *_args):
        pass
