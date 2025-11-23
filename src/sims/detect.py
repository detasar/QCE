from __future__ import annotations
import numpy as np
from typing import Tuple


def sample_clicks(p_true: float, eta: float, dark: float, rng: np.random.Generator | None = None) -> Tuple[bool, int]:
    """Return (clicked, observed_bit in {+1,-1}).
    With prob eta, a true detection occurs; outcome drawn from Bernoulli with P(+1)=p_true.
    With prob dark, a dark count may produce a random click if no true detection.
    If both occur, prefer true detection.
    """
    rng = rng or np.random.default_rng()
    true_det = rng.random() < eta
    if true_det:
        bit = 1 if (rng.random() < p_true) else -1
        return True, bit
    # no true detection; maybe dark
    if rng.random() < dark:
        bit = 1 if (rng.random() < 0.5) else -1
        return True, bit
    return False, 0


def post_select(maskA: np.ndarray, maskB: np.ndarray) -> np.ndarray:
    return maskA & maskB


def abstention_mask(maskA: np.ndarray, maskB: np.ndarray) -> np.ndarray:
    return ~(maskA & maskB)
