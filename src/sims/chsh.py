from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from .noise import apply_depolarizing, apply_axis_dephasing, jitter_angles


def ideal_correlation(theta_a: float, theta_b: float) -> float:
    """EPR-Bohm qubit correlation for measurement angles on Bloch equator: E = cos(2*(a-b))."""
    return float(np.cos(2.0 * (theta_a - theta_b)))


def sample_pair(E: float, rng: np.random.Generator) -> Tuple[int, int]:
    """Sample (x,y) in {+1,-1}^2 with zero marginals and correlation E.
    P(x=y) = (1+E)/2; draw x uniform, set y = x with prob (1+E)/2, else -x.
    """
    x = 1 if rng.random() < 0.5 else -1
    same = rng.random() < (1 + E) / 2
    y = x if same else -x
    return x, y


def estimate_chsh_S(x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    def E(ab0: int, ab1: int) -> float:
        mask = (a == ab0) & (b == ab1)
        if mask.sum() == 0:
            return 0.0
        return float((x[mask] * y[mask]).mean())
    S = E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1)
    return float(np.abs(S))


def make_settings(A0: float, A1: float, B0: float, B1: float):
    return (A0, A1, B0, B1)


def simulate_chsh(N: int,
                  angles: Tuple[float, float, float, float],
                  noise_cfg: Dict,
                  detect_cfg: Dict,
                  seed: int | None = None) -> Dict:
    rng = np.random.default_rng(seed)
    A0, A1, B0, B1 = angles
    # optional jitter/bias
    A0, A1, B0, B1 = jitter_angles((A0, A1, B0, B1),
                                   float(noise_cfg.get('jitter_sigma', 0.0)),
                                   float(noise_cfg.get('bias_delta', 0.0)), rng)

    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    x = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    clickA = np.zeros(N, dtype=bool)
    clickB = np.zeros(N, dtype=bool)

    etaA = float(detect_cfg.get('etaA', 1.0))
    etaB = float(detect_cfg.get('etaB', 1.0))
    darkA = float(detect_cfg.get('darkA', 0.0))
    darkB = float(detect_cfg.get('darkB', 0.0))

    for i in range(N):
        theta_a = A0 if a[i] == 0 else A1
        theta_b = B0 if b[i] == 0 else B1
        E = ideal_correlation(theta_a, theta_b)
        # white depolarization
        p_dep = float(noise_cfg.get('p_depol', 0.0))
        if p_dep:
            E = apply_depolarizing(E, p_dep)
        # axis dephasing: approximate by mapping a-context to X/Z
        px = float(noise_cfg.get('px', 0.0))
        pz = float(noise_cfg.get('pz', 0.0))
        axis = 'X' if a[i] == 0 else 'Z'
        if px or pz:
            E = apply_axis_dephasing(E, px, pz, axis)

        # draw pair with target correlation
        xi, yi = sample_pair(E, rng)
        # detection modeling: true detection keeps xi/yi; dark counts add random if no true detection
        trueA = (rng.random() < etaA)
        trueB = (rng.random() < etaB)
        if trueA:
            clickA[i] = True
            x[i] = xi
        else:
            if rng.random() < darkA:
                clickA[i] = True
                x[i] = 1 if (rng.random() < 0.5) else -1
        if trueB:
            clickB[i] = True
            y[i] = yi
        else:
            if rng.random() < darkB:
                clickB[i] = True
                y[i] = 1 if (rng.random() < 0.5) else -1

    S = estimate_chsh_S(x, y, a, b)
    return {
        'x': x, 'y': y, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'clickA': clickA,
            'clickB': clickB,
            'etaA': etaA, 'etaB': etaB, 'darkA': darkA, 'darkB': darkB,
        }
    }
