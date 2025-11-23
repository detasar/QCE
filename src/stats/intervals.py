from __future__ import annotations
from math import sqrt
from typing import Tuple
from scipy.stats import norm


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    radius = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return (lo, hi)
