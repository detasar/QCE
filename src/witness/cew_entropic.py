from __future__ import annotations
import numpy as np
from typing import Tuple


def entropic_bound(mu_C: float, sA_givenB: float = 0.0) -> float:
    return float(-2.0 * np.log2(mu_C) + sA_givenB)


def estimate_entropy_conditional(samples: np.ndarray, bins: int = 16, smooth: float = 1e-3) -> float:
    """Estimate H(X|B). samples: array of pairs (x,b) with discrete small alphabet.
    Here we treat x,b in {0,1} for simplicity; bins is ignored for discrete case.
    """
    x = samples[:, 0].astype(int)
    b = samples[:, 1].astype(int)
    H = 0.0
    for bv in [0, 1]:
        mask = (b == bv)
        n = int(mask.sum())
        if n == 0:
            continue
        cnt0 = (x[mask] == 0).sum()
        cnt1 = n - cnt0
        p0 = (cnt0 + smooth) / (n + 2 * smooth)
        p1 = (cnt1 + smooth) / (n + 2 * smooth)
        Hb = - (p0 * np.log2(p0) + p1 * np.log2(p1))
        H += (n / len(x)) * Hb
    return float(H)


def cew_e_decision(Hx_givenB: float, Hz_givenB: float, bound: float, alpha_band: float | None = None) -> int:
    return int((Hx_givenB + Hz_givenB) < bound)
