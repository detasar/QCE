from __future__ import annotations
import numpy as np
from typing import Tuple


def bootstrap_ci_mean(values, B: int = 2000, alpha: float = 0.05, rng: np.random.Generator | None = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng()
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boots[b] = values[idx].mean()
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return (float(lo), float(hi))
