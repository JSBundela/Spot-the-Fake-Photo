"""
Trains the real-vs-screen classifier on data/{real,screen}.

Feature selection: ranks the ~34 features in features.py by importance and
keeps only the top 10 (features.py has more detail); with 114 training
images, the full set overfits and the smaller subset generalizes better.

Model: ExtraTrees beat RandomForest/SVM/GBoost/logreg on repeated CV, so
that's what's shipped. Rather than pickling the sklearn model, this script
exports each tree's raw arrays (feature/threshold/children/leaf counts) as
plain numpy, reproduced by rf_light.py -- predict.py never imports
scikit-learn, which costs ~2s to import on its own.

Usage:
    python3 train.py
Writes model.pkl (scaler mean/scale, tree arrays, selected feature indices,
threshold).
"""
import glob
import os
import pickle

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from features import feature_vector, FEATURE_NAMES
from rf_light import forest_proba

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

N_ESTIMATORS = 400
MAX_DEPTH = 6
N_TOP_FEATURES = 10
CV_SEEDS = (1, 2, 3, 4, 5, 6, 7)


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

    ranker = RandomForestClassifier(n_estimators=500, max_depth=MAX_DEPTH, random_state=42)
    ranker.fit(StandardScaler().fit_transform(X), y)
    order = np.argsort(-ranker.feature_importances_)
    top_idx = order[:N_TOP_FEATURES]
    top_names = [FEATURE_NAMES[i] for i in top_idx]
    print(f"Top {N_TOP_FEATURES} features by importance: {top_names}\n")
    X_top = X[:, top_idx]

    print(f"Model comparison ({len(CV_SEEDS)}x5-fold repeated CV, {N_TOP_FEATURES}-feature subset):")
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm_rbf": SVC(kernel="rbf", C=32.0, gamma=0.05, class_weight="balanced"),
        "gboost": GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42),
    }
    best_name, best_acc = None, -1
    for name, clf in candidates.items():
        m, s = repeated_cv_accuracy(X_top, y, clf)
        print(f"  {name}: repeated cv accuracy = {m:.3f} +- {s:.3f}")
        if m > best_acc:
            best_acc, best_name = m, name
    print(f"  -> {best_name} selected (highest repeated-CV accuracy).\n")

    scaler_full = StandardScaler().fit(X_top)
    X_s = scaler_full.transform(X_top)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    et_for_cv = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
    oof_proba = cross_val_predict(et_for_cv, X_s, y, cv=skf, method="predict_proba")[:, 1]

    threshold = pick_threshold(y, oof_proba)
    y_pred = (oof_proba >= threshold).astype(int)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, oof_proba)
    cm = confusion_matrix(y, y_pred)

    print(f"Out-of-fold accuracy (single representative split) on all {len(y)} photos: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    print(classification_report(y, y_pred, target_names=["real", "screen"]))

    final_scaler = StandardScaler().fit(X_top)
    final_model = ExtraTreesClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42)
    final_model.fit(final_scaler.transform(X_top), y)

    trees = export_trees(final_model)

    X_check_s = final_scaler.transform(X_top[:20])
    sk_proba = final_model.predict_proba(X_check_s)[:, 1]
    manual_proba = np.array([forest_proba(x, trees)[1] for x in X_check_s])
    max_diff = np.max(np.abs(sk_proba - manual_proba))
    print(f"(sanity check: numpy vs sklearn forest predict_proba max abs diff = {max_diff:.2e})")
    assert max_diff < 1e-9, "numpy random forest does not match sklearn's"

    bundle = {
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
        "trees": trees,
        "feature_names": FEATURE_NAMES,
        "selected_feature_idx": top_idx.tolist(),
        "selected_feature_names": top_names,
        "threshold": float(threshold),
        "oof_accuracy": float(acc),
        "oof_auc": float(auc),
        "repeated_cv_accuracy_mean": best_acc,
        "model_name": best_name,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nSaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
