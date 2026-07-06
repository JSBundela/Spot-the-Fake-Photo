"""
Trains the shipped real-vs-screen classifier: a linear SVM on top of frozen
MobileNetV3-small (ImageNet-pretrained) embeddings (see cnn_embed.py and
models/mobilenet_v3_small_embedding.onnx). This beat the hand-crafted-feature
approach (train_handcrafted_features.py, 92.1% +/- 4.6%) on the same
evaluation methodology -- see EXPERIMENTS.md for the full comparison and why
a CNN was tried at all, and for the ML-runtime-dependency tradeoff this
approach accepts in exchange for the higher accuracy.

No feature selection here (all 576 embedding dimensions are used directly),
so there's no equivalent of the hand-crafted approach's nested-feature-
selection leakage. Reported accuracy uses each model's own standard decision
rule (sklearn's .predict(), not a threshold tuned on the evaluation data).
The probability calibration shipped in model.pkl (Platt scaling, fit on
cross-validated decision values, never on the fold being scored) is for
producing a well-behaved 0-1 score, not for picking a favorable threshold.

Usage:
    python3 train.py
Writes model.pkl (scaler mean/scale, SVM coefficients, sigmoid calibration
A/B, decision threshold in probability space).
"""
import glob
import os
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from cnn_embed import embed
from svm_light import linear_decision_function, sigmoid_proba

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

CV_SEEDS = (1, 2, 3, 4, 5, 6, 7)
CONFUSION_MATRIX_SEED = CV_SEEDS[0]


def load_folder(folder, label):
    import cv2
    X, y = [], []
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        img = cv2.imread(p)
        if img is None:
            continue
        rgb = img[:, :, ::-1]
        X.append(embed(rgb))
        y.append(label)
    return X, y


def load_phone_dataset():
    X, y = [], []
    for label, sub in ((0, "real"), (1, "screen")):
        xi, yi = load_folder(os.path.join(DATA_DIR, sub), label)
        X += xi; y += yi
    return np.array(X), np.array(y)


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


def fit_sigmoid_calibration(decision_values, y):
    """P(screen) = 1 / (1 + exp(A*f + B)), fit by 1-D logistic regression on
    cross-validated (not in-sample) decision values."""
    lr = LogisticRegression(max_iter=2000)
    lr.fit(decision_values.reshape(-1, 1), y)
    A = -float(lr.coef_[0, 0])
    B = -float(lr.intercept_[0])
    return A, B


def main():
    print("Extracting MobileNetV3-small embeddings for genuine phone photos...")
    X, y = load_phone_dataset()
    print(f"Phone dataset: {len(X)} images ({(y==0).sum()} real, {(y==1).sum()} screen), "
          f"embedding dim {X.shape[1]}\n")

    print(f"Model comparison ({len(CV_SEEDS)}x5-fold repeated CV, no feature selection needed):")
    candidates = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "svm_linear": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=400, max_depth=6, random_state=42),
    }
    best_name, best_acc = None, -1
    for name, clf in candidates.items():
        m, s = repeated_cv_accuracy(X, y, clf)
        print(f"  {name}: repeated cv accuracy = {m:.3f} +- {s:.3f}")
        if m > best_acc:
            best_acc, best_name = m, name
    print(f"  -> {best_name} selected (highest repeated-CV accuracy).\n")
    assert best_name == "svm_linear", (
        f"expected svm_linear to win (see EXPERIMENTS.md); got {best_name}. "
        "svm_light.py only supports a linear SVM -- update it if this changes."
    )

    # ---- Single representative split (reusing a seed already in the sweep), for a
    # concrete confusion matrix. Uses each model's own standard decision rule. ----
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFUSION_MATRIX_SEED)
    clf = SVC(kernel="linear", C=1.0, class_weight="balanced")
    y_pred = np.zeros(len(y), dtype=int)
    decision_oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf.fit(scaler.transform(X[tr]), y[tr])
        y_pred[te] = clf.predict(scaler.transform(X[te]))
        decision_oof[te] = clf.decision_function(scaler.transform(X[te]))

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, decision_oof)
    cm = confusion_matrix(y, y_pred)
    print(f"Out-of-fold accuracy (seed={CONFUSION_MATRIX_SEED}, standard decision rule, "
          f"all {len(y)} photos): {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(classification_report(y, y_pred, target_names=["real", "screen"]))

    # ---- Final shipped model: fit on ALL data ----
    final_scaler = StandardScaler().fit(X)
    X_s = final_scaler.transform(X)
    final_svm = SVC(kernel="linear", C=1.0, class_weight="balanced")
    final_svm.fit(X_s, y)

    # Probability calibration: fit on cross-validated (out-of-fold) decision
    # values from the final training data, not on the final model's own
    # in-sample decision values -- keeps the calibration itself leakage-free.
    cv_decision = cross_val_predict(
        SVC(kernel="linear", C=1.0, class_weight="balanced"),
        X_s, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFUSION_MATRIX_SEED),
        method="decision_function",
    )
    A, B = fit_sigmoid_calibration(cv_decision, y)
    # The shipped threshold corresponds to the SVM's own native decision
    # boundary (decision=0), expressed in probability space after
    # calibration -- not a threshold tuned to maximize accuracy.
    shipped_threshold = sigmoid_proba(0.0, A, B)

    coef = final_svm.coef_[0]
    intercept = float(final_svm.intercept_[0])

    manual_decision = np.array([linear_decision_function(x, coef, intercept) for x in X_s[:20]])
    sk_decision = final_svm.decision_function(X_s[:20])
    max_diff = np.max(np.abs(manual_decision - sk_decision))
    print(f"(sanity check: numpy vs sklearn linear decision function max abs diff = {max_diff:.2e})")
    assert max_diff < 1e-9, "numpy linear SVM does not match sklearn's"

    bundle = {
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
        "svm_coef": coef,
        "svm_intercept": intercept,
        "sigmoid_A": A,
        "sigmoid_B": B,
        "threshold": float(shipped_threshold),
        "oof_accuracy": float(acc),
        "oof_auc": float(auc),
        "repeated_cv_accuracy_mean": best_acc,
        "model_name": best_name,
        "embedding_model": "mobilenet_v3_small (ImageNet, frozen)",
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
