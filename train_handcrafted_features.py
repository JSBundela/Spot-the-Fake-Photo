"""
NOT the shipped model -- see train.py for that (CNN embeddings + linear SVM,
which scored higher). This is the earlier hand-crafted-feature approach,
kept for reference and comparison; EXPERIMENTS.md documents both. Writes to
model_handcrafted.pkl, not model.pkl, so it can't be confused with or
overwrite the shipped model.

Trains the real-vs-screen classifier on data/{real,screen} using ~34
hand-engineered features (features.py) reduced to the top 10 by importance.

Nested cross-validation: feature ranking/selection is redone inside every
training fold (using only that fold's training data), not once on the full
114 images -- doing it on the full dataset first and reusing the same
features for every fold leaks test-fold information into the feature choice
and inflates the reported accuracy. See EXPERIMENTS.md for the full
before/after comparison and other design decisions (hyperparameter tuning,
model comparison).

Reported accuracy uses a fixed 0.5 decision threshold (no threshold tuned on
the evaluation data itself).

Model: ExtraTrees beat RandomForest/SVM/GBoost/logreg on repeated nested CV.
Rather than pickling the sklearn model, this script exports each tree's raw
arrays (feature/threshold/children/leaf counts) as plain numpy, reproduced
by rf_light.py.

Usage:
    python3 train_handcrafted_features.py
Writes model_handcrafted.pkl (scaler mean/scale, tree arrays, selected
feature indices, threshold).
"""
import glob
import os
import pickle

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from features import feature_vector, FEATURE_NAMES
from rf_light import forest_proba

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_handcrafted.pkl")

N_ESTIMATORS = 400
MAX_DEPTH = 6
N_TOP_FEATURES = 10
CV_SEEDS = (1, 2, 3, 4, 5, 6, 7)
CONFUSION_MATRIX_SEED = CV_SEEDS[0]  # reuse a seed already in the sweep, not an arbitrary extra one


def load_folder(folder, label):
    X, y = [], []
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        img = cv2.imread(p)
        if img is None:
            continue
        vec, _ = feature_vector(img)
        X.append(vec)
        y.append(label)
    return X, y


def load_phone_dataset():
    X, y = [], []
    for label, sub in ((0, "real"), (1, "screen")):
        xi, yi = load_folder(os.path.join(DATA_DIR, sub), label)
        X += xi; y += yi
    return np.array(X), np.array(y)


def select_top_features(X_train, y_train, n_top=N_TOP_FEATURES):
    """Ranks features by importance using only X_train/y_train, returns the top indices."""
    ranker = RandomForestClassifier(n_estimators=500, max_depth=MAX_DEPTH, random_state=42)
    ranker.fit(StandardScaler().fit_transform(X_train), y_train)
    order = np.argsort(-ranker.feature_importances_)
    return order[:n_top]


def pick_threshold(y_true, scores):
    best_t, best_j = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        pred = (scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def nested_cv_accuracy(X, y, clf, seeds=CV_SEEDS, n_top=N_TOP_FEATURES):
    """Repeated stratified k-fold CV with feature selection redone inside each
    training fold, and a fixed 0.5 threshold -- no information from a test
    fold ever touches feature selection, scaling, or thresholding."""
    accs = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            top_idx = select_top_features(X[tr], y[tr], n_top)
            X_tr, X_te = X[tr][:, top_idx], X[te][:, top_idx]
            scaler = StandardScaler().fit(X_tr)
            clf.fit(scaler.transform(X_tr), y[tr])
            pred = clf.predict(scaler.transform(X_te))
            accs.append((pred == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def nested_cv_oof_predictions(X, y, clf, seed, n_top=N_TOP_FEATURES):
    """One seed's 5-fold CV, nested feature selection, returns per-image
    out-of-fold probabilities (every image scored by a model that never saw
    it, with features chosen only from that fold's training data)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_proba = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        top_idx = select_top_features(X[tr], y[tr], n_top)
        X_tr, X_te = X[tr][:, top_idx], X[te][:, top_idx]
        scaler = StandardScaler().fit(X_tr)
        clf.fit(scaler.transform(X_tr), y[tr])
        oof_proba[te] = clf.predict_proba(scaler.transform(X_te))[:, 1]
    return oof_proba


def export_trees(model):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append((
            t.feature.copy(),
            t.threshold.astype(np.float64).copy(),
            t.children_left.copy(),
            t.children_right.copy(),
            t.value[:, 0, :].copy(),
        ))
    return trees


def main():
    print("Extracting features from genuine phone photos...")
    X, y = load_phone_dataset()
    print(f"Phone dataset: {len(X)} images ({(y==0).sum()} real, {(y==1).sum()} screen)\n")

    print(f"Model comparison ({len(CV_SEEDS)}x5-fold nested CV, feature selection redone per fold, 0.5 threshold):")
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", C=32.0, gamma=0.05, class_weight="balanced"),
        "gboost": GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
    }
    best_name, best_acc = None, -1
    for name, clf in candidates.items():
        m, s = nested_cv_accuracy(X, y, clf)
        print(f"  {name}: nested cv accuracy = {m:.3f} +- {s:.3f}")
        if m > best_acc:
            best_acc, best_name = m, name
    print(f"  -> {best_name} selected (highest nested-CV accuracy).\n")

    best_clf_factory = {
        "logreg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm_rbf": lambda: SVC(kernel="rbf", C=32.0, gamma=0.05, class_weight="balanced", probability=True),
        "gboost": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=42),
        "random_forest": lambda: RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
    }[best_name]

    oof_proba = nested_cv_oof_predictions(X, y, best_clf_factory(), seed=CONFUSION_MATRIX_SEED)
    y_pred = (oof_proba >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, oof_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"Out-of-fold accuracy (seed={CONFUSION_MATRIX_SEED}, nested feature selection, 0.5 threshold, "
          f"all {len(y)} photos): {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(classification_report(y, y_pred, target_names=["real", "screen"]))

    # ---- Final shipped model: fit feature selection + model on ALL data ----
    top_idx = select_top_features(X, y, N_TOP_FEATURES)
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    print(f"Final feature selection (fit on all {len(y)} images, for the shipped model): {top_names}")
    X_top = X[:, top_idx]

    final_scaler = StandardScaler().fit(X_top)
    final_model = best_clf_factory()
    final_model.fit(final_scaler.transform(X_top), y)

    # Threshold for the SHIPPED model: chosen via Youden's J on the full training
    # set. This is a deliberate operating-point choice for production (see
    # report.md), kept separate from the accuracy figures above, which use a
    # fixed 0.5 threshold precisely so this choice can't inflate them.
    full_proba = final_model.predict_proba(final_scaler.transform(X_top))[:, 1]
    shipped_threshold = pick_threshold(y, full_proba)

    if best_name == "extra_trees":
        trees = export_trees(final_model)
        X_check_s = final_scaler.transform(X_top[:20])
        sk_proba = final_model.predict_proba(X_check_s)[:, 1]
        manual_proba = np.array([forest_proba(x, trees)[1] for x in X_check_s])
        max_diff = np.max(np.abs(sk_proba - manual_proba))
        print(f"(sanity check: numpy vs sklearn forest predict_proba max abs diff = {max_diff:.2e})")
        assert max_diff < 1e-9, "numpy random forest does not match sklearn's"
    else:
        raise RuntimeError(
            f"{best_name} won nested CV but rf_light.py only supports RandomForest/ExtraTrees -- "
            "predict.py would need a different numpy-only export for this model type."
        )

    bundle = {
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
        "trees": trees,
        "feature_names": FEATURE_NAMES,
        "selected_feature_idx": top_idx.tolist(),
        "selected_feature_names": top_names,
        "threshold": float(shipped_threshold),
        "oof_accuracy": float(acc),
        "oof_auc": float(auc),
        "nested_cv_accuracy_mean": best_acc,
        "model_name": best_name,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
