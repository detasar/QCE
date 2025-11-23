from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def frequency_model(features: np.ndarray, labels: np.ndarray) -> Dict[Any, Dict[int, float]]:
    """Return P(y|feature) via Laplace-smoothed counts for binary labels in {+1,-1}.
    Returns dict: feature -> {+1: p, -1: 1-p}
    """
    probs: Dict[Any, Dict[int, float]] = {}
    for f in np.unique(features):
        mask = (features == f)
        ys = labels[mask]
        npos = int((ys == 1).sum())
        n = int(mask.sum())
        # Laplace smoothing
        p = (npos + 1) / (n + 2)
        probs[f] = {+1: p, -1: 1 - p}
    return probs


def nonconformity_from_probs(p_hat: float) -> float:
    return 1.0 - p_hat


def knn_weighted_probs(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, K: int = 11, h: float = 0.25) -> np.ndarray:
    """Gaussian-weighted kNN probability estimates P(y=+1|x).
    X_* shape: (n, d); y_train in {+1,-1}.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    ntr = X_train.shape[0]
    out = np.empty(X_query.shape[0], dtype=float)
    for i, xq in enumerate(X_query):
        d2 = np.sum((X_train - xq) ** 2, axis=1)
        nn_idx = np.argsort(d2)[: max(1, min(K, ntr))]
        w = np.exp(-d2[nn_idx] / (2 * (h ** 2) + 1e-12))
        if w.sum() <= 0:
            out[i] = 0.5
        else:
            pos = (y_train[nn_idx] == 1).astype(float)
            out[i] = float(np.sum(w * pos) / np.sum(w))
    return out


def fit_platt(raw_p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit Platt scaling on raw probabilities via logistic regression on logit(p).
    Returns coefficients (A, B) such that calibrated p = sigmoid(A*logit(p)+B).
    """
    eps = 1e-6
    p = np.clip(raw_p, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    # Map labels {+1,-1} -> {1,0}
    y01 = (y == 1).astype(int)
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(logit, y01)
    A = float(lr.coef_.ravel()[0])
    B = float(lr.intercept_.ravel()[0])
    return A, B


def apply_platt(raw_p: np.ndarray, AB: Tuple[float, float]) -> np.ndarray:
    eps = 1e-6
    p = np.clip(raw_p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    A, B = AB
    z = A * logit + B
    return 1.0 / (1.0 + np.exp(-z))


def fit_isotonic(raw_p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression mapping raw_p -> calibrated probability."""
    ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    y01 = (y == 1).astype(int)
    ir.fit(raw_p, y01)
    return ir
