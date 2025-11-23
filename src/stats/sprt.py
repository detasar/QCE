from __future__ import annotations
import numpy as np
from typing import Tuple


def sprt_boundaries(p0: float, p1: float, alpha: float = 0.05, beta: float = 0.2) -> Tuple[float, float]:
    A = np.log((1 - beta) / alpha)
    B = np.log(beta / (1 - alpha))
    return float(A), float(B)


def sprt_update(logLR: float, x: int, p0: float, p1: float) -> float:
    inc = np.log((p1 if x else (1 - p1)) / (p0 if x else (1 - p0)))
    return float(logLR + inc)
