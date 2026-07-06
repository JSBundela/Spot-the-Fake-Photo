#!/usr/bin/env python3
"""
One-line predictor: is this a REAL photo or a PHOTO OF A SCREEN?

Usage:
    python3 predict.py path/to/image.jpg
    -> prints a single float in [0, 1]: 0 = real photo, 1 = photo of a screen.

Loads model.pkl (produced by train.py): a StandardScaler + linear SVM over
576-dim frozen MobileNetV3-small embeddings (cnn_embed.py, running on
onnxruntime -- not torch, which would conflict with this project's numpy
version; onnxruntime is also the appropriate lightweight runtime for
on-device use, unlike bundling all of PyTorch). See EXPERIMENTS.md for why
this replaced an earlier hand-crafted-feature model.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from cnn_embed import embed
from svm_light import linear_decision_function, sigmoid_proba

MODEL_PATH = Path(__file__).parent / "model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(bgr, bundle):
    rgb = bgr[:, :, ::-1]
    vec = embed(rgb)
    vec_s = (vec - bundle["scaler_mean"]) / bundle["scaler_scale"]
    decision = linear_decision_function(vec_s, bundle["svm_coef"], bundle["svm_intercept"])
    proba = sigmoid_proba(decision, bundle["sigmoid_A"], bundle["sigmoid_B"])
    return float(proba)


def main():
    parser = argparse.ArgumentParser(description="Score an image as real (0) vs photo-of-a-screen (1).")
    parser.add_argument("image_path")
    parser.add_argument("--timing", action="store_true", help="print latency to stderr")
    args = parser.parse_args()

    bundle = load_model()

    t0 = time.perf_counter()
    try:
        img = Image.open(args.image_path).convert("RGB")
    except Exception:
        print(f"error: could not read image at {args.image_path}", file=sys.stderr)
        sys.exit(1)
    bgr = np.array(img)[:, :, ::-1]

    score = predict(bgr, bundle)
    t1 = time.perf_counter()

    print(f"{score:.4f}")
    if args.timing:
        print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
