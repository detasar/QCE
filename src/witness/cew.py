from __future__ import annotations
import ast
import operator as op
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from ..stats.intervals import wilson_interval
from .cew_entropic import entropic_bound


def cardinality_product(card_X: float, card_Z: float) -> float:
    return float(card_X * card_Z)


def mean_product_over_runs(prod_list) -> float:
    arr = np.asarray(prod_list, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float('nan')


def _safe_eval(expr: str, variables: Dict[str, float]) -> float:
    """Safely evaluate arithmetic expression with given variables."""
    allowed_operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
                         ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise ValueError(f"Unknown variable {node.id}")
        if isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_operators[type(node.op)](_eval(node.operand))
        raise TypeError(node)

    tree = ast.parse(expr, mode='eval')
    return float(_eval(tree.body))


def threshold_L(alpha: float, c: float, cfg: Dict[str, Any]) -> float:
    expr = cfg.get('L_formula', '(1-alpha)**2 / c')
    val = _safe_eval(expr, {'alpha': float(alpha), 'c': float(c)})
    scale = float(cfg.get('scale', 1.0))
    return float(scale * val)


def threshold_L_entropic(alpha: float, mu_C: float, sA_givenB: float = 0.0,
                         tau: float = 1.0, gamma: float = 1.0) -> float:
    """Map entropic EUR bound to CEW product threshold.
    Heuristic: prediction-set cardinality ~ 2^{H(.|B)} scaled by tau(α).
    Then L = gamma * (tau^2) * 2^{H_sum_bound}, H_sum_bound = -2 log2 mu_C + sA|B.
    """
    Hsum = entropic_bound(mu_C, sA_givenB)
    return float(gamma * (tau ** 2) * (2.0 ** Hsum))


def threshold_L_s(S_value: float, a: float = 1.2, b: float = -0.1) -> float:
    """S-based threshold: L = a + b*(S-2). Decrease with stronger S."""
    return float(a + b * (S_value - 2.0))


def threshold_L_eta(etaA: float, etaB: float, L0: float = 1.2, beta: float = 1.0) -> float:
    """Efficiency-based threshold: higher efficiency reduces threshold.
    L = L0 / (min(etaA,etaB)^beta)."""
    m = max(min(etaA, etaB), 1e-6)
    return float(L0 / (m ** beta))


def threshold_L_quantile(prod_samples, q: float = 0.1) -> float:
    import numpy as np
    arr = np.asarray(prod_samples, dtype=float)
    if arr.size == 0:
        return float('nan')
    return float(np.quantile(arr, q))


def threshold_L_cpcalib(tauX: float, tauZ: float, gamma: float = 1.0) -> float:
    return float(gamma * tauX * tauZ)


def tau_cp(alpha: float, K: int = 2) -> float:
    """Heuristic CP expected set size under perfect calibration: E|C| ≈ 1 + (K-1)·alpha.
    For binary (K=2), tau(α)=1+α.
    """
    return float(1.0 + (K - 1) * alpha)


def evaluate_cew_over_grid(records: list[dict], alpha: float, c: float, cfg: Dict[str, Any] | None = None) -> Tuple[pd.DataFrame, float]:
    """records: list of dicts with keys x, y, avg_card_X, avg_card_Z, S_mean (optional)
    Returns dataframe with CEW_rate and Wilson CI columns.
    """
    L = threshold_L(alpha, c, cfg or {'L_formula': '(1-alpha)**2 / c'})
    rows = []
    cew_flags = []
    for r in records:
        prod_mean = cardinality_product(r['avg_card_X'], r['avg_card_Z'])
        cew = int(prod_mean < L)
        cew_flags.append(cew)
        rows.append({
            'x': r['x'], 'y': r['y'],
            'prod_mean': prod_mean,
            'S_mean': r.get('S_mean', np.nan),
            'CEW': cew
        })
    df = pd.DataFrame(rows)
    # aggregate by (x,y)
    grouped = df.groupby(['x', 'y'])
    out_rows = []
    for (gx, gy), g in grouped:
        k = int(g['CEW'].sum())
        n = int(len(g))
        lo, hi = wilson_interval(k, n, alpha=0.05)
        out_rows.append({'x': gx, 'y': gy, 'CEW_rate': k / n if n else np.nan, 'CEW_lo': lo, 'CEW_hi': hi,
                         'S_mean': g['S_mean'].mean(), 'n_rep': n})
    return pd.DataFrame(out_rows), L
