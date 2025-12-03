"""
Data Utilities for TARA Quantum Certification

Functions for loading, processing, and manipulating quantum data.

Author: Davut Emre Tasar
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def load_ibm_data(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load IBM quantum data from CSV.

    Args:
        filepath: Path to CSV file with columns x, y, a, b

    Returns:
        Dict with 'x', 'y', 'a', 'b' numpy arrays (int64)
    """
    df = pd.read_csv(filepath)
    return {
        'x': df['x'].values.astype(np.int64),
        'y': df['y'].values.astype(np.int64),
        'a': df['a'].values.astype(np.int64),
        'b': df['b'].values.astype(np.int64)
    }


def compute_chsh(data: Dict[str, np.ndarray]) -> float:
    """
    Compute CHSH S-value from quantum data.

    S = E(0,0) + E(0,1) + E(1,0) - E(1,1)
    where E(x,y) = <a*b> with a,b in {-1,+1}

    Args:
        data: Dict with 'x', 'y', 'a', 'b' arrays

    Returns:
        CHSH S-value (classical limit: 2, quantum max: 2*sqrt(2) = 2.83)
    """
    x, y, a, b = data['x'], data['y'], data['a'], data['b']

    # Convert to +/-1
    a_pm = 2 * a - 1
    b_pm = 2 * b - 1

    E = {}
    for xi in [0, 1]:
        for yi in [0, 1]:
            mask = (x == xi) & (y == yi)
            if mask.sum() > 0:
                E[(xi, yi)] = (a_pm[mask] * b_pm[mask]).mean()
            else:
                E[(xi, yi)] = 0.0

    return E[(0,0)] + E[(0,1)] + E[(1,0)] - E[(1,1)]


def compute_setting_distribution(data: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], float]:
    """
    Compute the distribution of settings (x, y).

    Args:
        data: Dict with 'x', 'y' arrays

    Returns:
        Dict mapping (x, y) to fraction of total samples
    """
    n = len(data['x'])
    dist = {}

    for xi in [0, 1]:
        for yi in [0, 1]:
            count = np.sum((data['x'] == xi) & (data['y'] == yi))
            dist[(xi, yi)] = count / n

    return dist


def compute_conditional_probs(data: Dict[str, np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Compute P(a, b | x, y) for each setting.

    Args:
        data: Dict with 'x', 'y', 'a', 'b' arrays

    Returns:
        Dict mapping (x, y) to 2x2 probability matrix P[a, b]
    """
    x, y, a, b = data['x'], data['y'], data['a'], data['b']
    probs = {}

    for xi in [0, 1]:
        for yi in [0, 1]:
            mask = (x == xi) & (y == yi)
            if mask.sum() > 0:
                counts = np.zeros((2, 2))
                for ai in [0, 1]:
                    for bi in [0, 1]:
                        counts[ai, bi] = np.sum((a[mask] == ai) & (b[mask] == bi))
                counts += 1e-10  # Avoid division by zero
                probs[(xi, yi)] = counts / counts.sum()
            else:
                probs[(xi, yi)] = np.ones((2, 2)) / 4

    return probs


def split_data(data: Dict[str, np.ndarray],
               train_ratio: float = 0.5,
               shuffle: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into two parts (e.g., calibration and test).

    Args:
        data: Dict with arrays
        train_ratio: Fraction for first split
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (first_split, second_split)
    """
    n = len(data['x'])

    if shuffle:
        perm = np.random.permutation(n)
        data = {k: v[perm] for k, v in data.items()}

    n_train = int(n * train_ratio)

    first = {k: v[:n_train] for k, v in data.items()}
    second = {k: v[n_train:] for k, v in data.items()}

    return first, second


def create_interpolated_data(
    honest_data: Dict[str, np.ndarray],
    eve_data: Dict[str, np.ndarray],
    alpha: float
) -> Dict[str, np.ndarray]:
    """
    Create interpolated mixture of honest and Eve data.

    alpha = 1.0: 100% honest
    alpha = 0.0: 100% Eve
    alpha = 0.5: 50% honest, 50% Eve

    Args:
        honest_data: Honest (IBM) data dictionary.
        eve_data: Eve-generated data dictionary.
        alpha: Mixing ratio (1.0 = all honest, 0.0 = all Eve).

    Returns:
        Mixed data dictionary (shuffled).
    """
    n = min(len(honest_data['x']), len(eve_data['x']))
    n_honest = int(n * alpha)
    n_eve = n - n_honest

    honest_idx = np.random.choice(len(honest_data['x']), n_honest, replace=False)
    eve_idx = np.random.choice(len(eve_data['x']), n_eve, replace=False)

    mixed = {
        'x': np.concatenate([honest_data['x'][honest_idx], eve_data['x'][eve_idx]]),
        'y': np.concatenate([honest_data['y'][honest_idx], eve_data['y'][eve_idx]]),
        'a': np.concatenate([honest_data['a'][honest_idx], eve_data['a'][eve_idx]]),
        'b': np.concatenate([honest_data['b'][honest_idx], eve_data['b'][eve_idx]]),
    }

    # Shuffle
    perm = np.random.permutation(n)
    return {k: v[perm] for k, v in mixed.items()}
