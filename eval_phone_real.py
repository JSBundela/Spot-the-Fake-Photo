"""
Evaluates model.pkl against data/{real,screen} (converted from
data/REAL_DATA_captured_by_phone). This is in-sample accuracy (the model was
trained on this same data) -- a fit sanity check, not the accuracy to
report. The honest, no-leakage number is train.py's out-of-fold accuracy,
saved into model.pkl as "oof_accuracy".
"""
import glob
import os

import cv2

from predict import load_model, predict

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    bundle = load_model()
    threshold = bundle["threshold"]
    total_correct, total_n = 0, 0

    for label, subdir in ((0, "real"), (1, "screen")):
        paths = sorted(glob.glob(os.path.join(DATA_DIR, subdir, "*.jpg")))
        correct = 0
        scores = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"  could not read {p}")
                continue
            score = predict(img, bundle)
            scores.append(score)
            pred_label = int(score >= threshold)
            correct += pred_label == label
        print(f"--- {subdir} ({len(paths)} images) --- accuracy: {correct}/{len(paths)} = {correct/len(paths):.1%}")
        print(f"  score distribution: min={min(scores):.3f} max={max(scores):.3f} mean={sum(scores)/len(scores):.3f}")
        total_correct += correct
        total_n += len(paths)

    print(f"\nOverall in-sample accuracy: {total_correct}/{total_n} = {total_correct/total_n:.1%} "
          f"(threshold={threshold:.2f})")
    print("(out-of-fold accuracy, the honest number, is printed by train.py / model.pkl['oof_accuracy'])")


if __name__ == "__main__":
    main()
