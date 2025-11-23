from __future__ import annotations
import numpy as np


def selective_coverage(y_true: np.ndarray, in_set_mask: np.ndarray, abstain_mask: np.ndarray) -> float:
    """Compute coverage P(Y in C | abstain=False)."""
    valid = ~abstain_mask
    if valid.sum() == 0:
        return float('nan')
    return float((in_set_mask[valid]).mean())
