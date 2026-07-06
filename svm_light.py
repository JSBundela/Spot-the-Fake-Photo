"""
Numpy-only linear SVM decision function + sigmoid calibration, so predict.py
doesn't need scikit-learn just to compute a dot product. train.py exports
the fitted coefficients; this reproduces sklearn's decision_function exactly.
"""
import numpy as np


def linear_decision_function(x, coef, intercept):
    return float(np.dot(x, coef) + intercept)


def sigmoid_proba(decision, A, B):
    return 1.0 / (1.0 + np.exp(A * decision + B))
