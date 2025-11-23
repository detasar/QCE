from __future__ import annotations
import numpy as np
from typing import Dict, Iterable


def group_keys(contexts: np.ndarray) -> np.ndarray:
    return contexts.astype(int)


def groupwise_indices(groups: np.ndarray) -> Dict[int, np.ndarray]:
    """Map group value -> indices."""
    out: Dict[int, np.ndarray] = {}
    for g in np.unique(groups):
        out[int(g)] = np.flatnonzero(groups == g)
    return out
