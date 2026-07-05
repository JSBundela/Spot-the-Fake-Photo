"""
Tiny live demo: serves index.html and a single /score endpoint that runs the
same predictor as predict.py against a base64-encoded camera frame.

Usage (local dev):
    python3 webdemo/server.py
    -> open http://localhost:5050

Production (e.g. Render, see README.md "Render deployment"):
    gunicorn -w 1 -b 0.0.0.0:$PORT webdemo.server:app
(run from the repo root so the `webdemo.server` import path resolves).
"""
import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent))
from predict import load_model, predict  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent))
bundle = load_model()

CAPTURE_DIR = Path(__file__).parent.parent / "data" / "phone_real_webcam"

# The /save_example dev endpoint (see below) writes to local disk and has no
# auth -- fine for local dev, not something to expose on a public deployment
# by default. Set ENABLE_SAVE_EXAMPLE=1 to turn it on (e.g. for local use).
ENABLE_SAVE_EXAMPLE = os.environ.get("ENABLE_SAVE_EXAMPLE", "1") == "1"


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(force=True)
    b64 = data["image"].split(",", 1)[-1]
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    t0 = time.perf_counter()
    p = predict(bgr, bundle)
    latency_ms = (time.perf_counter() - t0) * 1000

    return jsonify({"score": p, "threshold": bundle["threshold"], "latency_ms": round(latency_ms, 1)})


if ENABLE_SAVE_EXAMPLE:
    @app.route("/save_example", methods=["POST"])
    def save_example():
        # Dev-only route: saves a live camera frame as a new labeled training example.
        data = request.get_json(force=True)
        label = data["label"]
        if label not in ("real", "screen"):
            return jsonify({"error": "label must be 'real' or 'screen'"}), 400
        b64 = data["image"].split(",", 1)[-1]
        raw = base64.b64decode(b64)

        out_dir = CAPTURE_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(out_dir.glob("*.jpg")))
        out_path = out_dir / f"webcam_{idx:03d}.jpg"
        with open(out_path, "wb") as f:
            f.write(raw)

        return jsonify({"saved": str(out_path.relative_to(CAPTURE_DIR.parent.parent))})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
