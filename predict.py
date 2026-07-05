#!/usr/bin/env python3
"""
One-line predictor: is this a REAL photo or a PHOTO OF A SCREEN?

Usage:
    python3 predict.py path/to/image.jpg
    -> prints a single float in [0, 1]: 0 = real photo, 1 = photo of a screen.

Loads model.pkl (produced by train.py): a StandardScaler + small ExtraTrees
forest over 10 of the ~34 features in features.py, stored as plain numpy
arrays. No scikit-learn import here -- that alone costs ~2s vs ~10ms for
feature extraction + tree evaluation (see rf_light.py).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2

from features import feature_vector
from rf_light import forest_proba

MODEL_PATH = Path(__file__).parent / "model.pkl"


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(bgr, bundle):
    vec, _ = feature_vector(bgr)
    vec = vec[bundle["selected_feature_idx"]] if "selected_feature_idx" in bundle else vec
    vec_s = (vec - bundle["scaler_mean"]) / bundle["scaler_scale"]
    proba = forest_proba(vec_s, bundle["trees"])
    return float(proba[1])


def main():
    parser = argparse.ArgumentParser(description="Score an image as real (0) vs photo-of-a-screen (1).")
    parser.add_argument("image_path")
    parser.add_argument("--timing", action="store_true", help="print latency to stderr")
    args = parser.parse_args()

    bundle = load_model()

    t0 = time.perf_counter()
    bgr = cv2.imread(args.image_path)
    if bgr is None:
        print(f"error: could not read image at {args.image_path}", file=sys.stderr)
        sys.exit(1)

    score = predict(bgr, bundle)
    t1 = time.perf_counter()

    print(f"{score:.4f}")
    if args.timing:
        print(f"latency: {(t1 - t0) * 1000:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
