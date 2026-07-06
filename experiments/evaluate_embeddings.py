"""
Evaluates classifiers on the frozen CNN embeddings saved by
extract_embeddings.py, using the same repeated stratified CV as train.py.
Run in the main project's venv (needs scikit-learn/numpy>=2, not torch).

Usage:
    python3 experiments/extract_embeddings.py   # separate venv, see requirements.txt
    python3 experiments/evaluate_embeddings.py  # main project venv
"""
import os

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npz")
CV_SEEDS = (1, 2, 3, 4, 5, 6, 7)


def repeated_cv_accuracy(X, y, clf, seeds=CV_SEEDS):
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler().fit(X[tr])
            clf.fit(scaler.transform(X[tr]), y[tr])
            pred = clf.predict(scaler.transform(X[te]))
            accs.append((pred == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    data = np.load(EMBEDDINGS_PATH)
    X, y = data["X"], data["y"]
    print(f"Loaded embeddings: {X.shape}\n")

    candidates = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "svm_linear": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, max_depth=6, random_state=42),
    }
    for name, clf in candidates.items():
        m, s = repeated_cv_accuracy(X, y, clf)
        print(f"  {name}: repeated cv accuracy = {m:.3f} +- {s:.3f}")


if __name__ == "__main__":
    main()
