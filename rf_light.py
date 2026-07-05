"""
Numpy-only forest predict_proba (works for RandomForest and ExtraTrees --
same tree_ structure). Avoids importing scikit-learn in predict.py just to
walk a few hundred small trees. train.py exports each tree's raw arrays;
this reproduces sklearn's predict_proba exactly.
"""
import numpy as np


def tree_proba(x, feature, threshold, children_left, children_right, value):
    node = 0
    while children_left[node] != -1:
        node = children_left[node] if x[feature[node]] <= threshold[node] else children_right[node]
    row = value[node]
    return row / row.sum()


def forest_proba(x, trees):
    """trees: list of (feature, threshold, children_left, children_right, value) tuples."""
    probs = np.array([tree_proba(x, *t) for t in trees])
    return probs.mean(axis=0)
