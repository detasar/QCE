from __future__ import annotations
import numpy as np
from typing import Tuple


def fit_gamma_cp(products: np.ndarray, tauX: np.ndarray, tauZ: np.ndarray, target_q: float = 0.1,
                 tol: float = 1e-3, max_iter: int = 50) -> float:
    """Calibrate gamma so that P(prod < gamma * tauX * tauZ) ≈ target_q under classical region.
    Uses bisection on gamma.
    """
    products = np.asarray(products, dtype=float)
    base = np.asarray(tauX, dtype=float) * np.asarray(tauZ, dtype=float)
    base = np.maximum(base, 1e-9)
    lo, hi = 0.1, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(products < mid * base))
        if rate > target_q:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol:
            break
    return float(0.5 * (lo + hi))


def fit_s_params(S: np.ndarray, products: np.ndarray) -> Tuple[float, float]:
    """Fit a,b in L(S)=a+b*(S-2) via linear regression to approximate prod trend.
    Returns (a,b)."""
    S = np.asarray(S, dtype=float)
    y = np.asarray(products, dtype=float)
    X = np.stack([np.ones_like(S), S - 2.0], axis=1)
    # least squares
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = theta.tolist()
    return float(a), float(b)


def fit_eta_params(eta: np.ndarray, products: np.ndarray) -> Tuple[float, float]:
    """Fit L(eta) = L0 / (min_eta^beta). Here eta is scalar per replicate (symmetrized).
    Solve log L ≈ log L0 - beta log eta via regression using product as proxy for L.
    """
    e = np.asarray(eta, dtype=float)
    y = np.maximum(np.asarray(products, dtype=float), 1e-9)
    X = np.stack([np.ones_like(e), -np.log(np.maximum(e, 1e-6))], axis=1)
    theta, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    logL0, beta = theta.tolist()
    L0 = np.exp(logL0)
    return float(L0), float(beta)
