from __future__ import annotations
from math import sqrt
from typing import Tuple
from scipy.stats import norm


def n_for_proportions(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approx. sample size for testing H0:p=p0 vs H1:p=p1 (two-sided), using normal approx.
    Returns minimal n."""
    z1 = norm.ppf(1 - alpha / 2)
    z2 = norm.ppf(power)
    var = p0 * (1 - p0) + p1 * (1 - p1)
    n = ((z1 * sqrt(p0 * (1 - p0)) + z2 * sqrt(p1 * (1 - p1))) ** 2) / ((p1 - p0) ** 2 + 1e-12)
    return int(max(1, round(n)))
