from __future__ import annotations
import numpy as np
from typing import Tuple


def benjamini_hochberg(pvals, q: float = 0.1) -> Tuple[np.ndarray, float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    if passed.any():
        k = np.where(passed)[0].max() + 1
        cutoff = ranked[k - 1]
    else:
        cutoff = 0.0
    reject = p <= cutoff
    return reject, float(cutoff)
