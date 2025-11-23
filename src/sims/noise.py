from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def apply_depolarizing(corr: float, p: float) -> float:
    """Depolarizing noise reduces correlation by visibility (1-2p) approx for binary outcomes.
    Here we use a visibility v = (1 - p), i.e., E -> (1-p)*E for simplicity.
    """
    v = max(0.0, 1.0 - p)
    return v * corr


def apply_axis_dephasing(corr: float, px: float, pz: float, axis: str) -> float:
    """Axis-dependent dephasing: reduce correlation more strongly on matching axis.
    axis in {"X","Z"}."""
    if axis.upper() == "X":
        v = max(0.0, 1.0 - px)
    else:
        v = max(0.0, 1.0 - pz)
    return v * corr


def jitter_angles(angles: Tuple[float, float, float, float], sigma: float, bias: float = 0.0,
                  rng: np.random.Generator | None = None) -> Tuple[float, float, float, float]:
    rng = rng or np.random.default_rng()
    if sigma <= 0 and bias == 0:
        return angles
    a0, a1, b0, b1 = angles
    j = rng.normal(0.0, sigma, size=4)
    return (a0 + bias + j[0], a1 + bias + j[1], b0 + bias + j[2], b1 + bias + j[3])
