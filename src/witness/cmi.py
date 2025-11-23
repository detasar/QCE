from __future__ import annotations
import numpy as np
from typing import Dict, List
from scipy import stats


def tv_distance_to_uniform(pvals: np.ndarray, bins: int = 20) -> float:
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return float('nan')
    hist, _ = np.histogram(pvals, bins=bins, range=(0.0, 1.0), density=True)
    tv = 0.5 * np.sum(np.abs(hist - 1.0) * (1.0 / bins))
    return float(tv)


def ks_ad_tests(pvals: np.ndarray) -> dict:
    if len(pvals) == 0:
        return {'ks_p': np.nan, 'ad_stat': np.nan}
    ks = stats.kstest(pvals, 'uniform', args=(0, 1))
    try:
        ad = stats.anderson(pvals, dist='uniform')
        ad_stat = ad.statistic
    except Exception:
        ad_stat = np.nan
    return {'ks_p': float(ks.pvalue), 'ad_stat': float(ad_stat)}


def simultaneous_calibration(pvals_by_context: Dict[int, np.ndarray], q: float = 0.1) -> dict:
    from ..stats.fdr import benjamini_hochberg
    pvals = []
    for ctx, arr in pvals_by_context.items():
        m = len(arr)
        if m == 0:
            continue
        # one-sample KS p-value per context
        ks = stats.kstest(arr, 'uniform', args=(0, 1))
        pvals.append(ks.pvalue)
    if not pvals:
        return {'reject': False, 'cutoff': np.nan}
    reject_mask, cutoff = benjamini_hochberg(pvals, q=q)
    return {'reject': bool(np.any(reject_mask)), 'cutoff': float(cutoff)}


def cmi_index(pvals_by_context: Dict[int, np.ndarray], weights: Dict[int, float] | None = None) -> float:
    weights = weights or {ctx: 1.0 for ctx in pvals_by_context.keys()}
    tvs = []
    ws = []
    for ctx, arr in pvals_by_context.items():
        tv = tv_distance_to_uniform(arr)
        w = weights.get(ctx, 1.0)
        if not np.isnan(tv):
            tvs.append(tv)
            ws.append(w)
    if not tvs:
        return float('nan')
    tvs = np.asarray(tvs)
    ws = np.asarray(ws)
    return float(np.average(tvs, weights=ws))


def ece_uniform(pvals: np.ndarray, bins: int = 20) -> float:
    """Expected calibration error vs Uniform(0,1): sum |p_i - u_i|/m; here histogram L1 vs 1/bins.
    Returns simple L1 histogram discrepancy (same as TV*2)."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.size == 0:
        return float('nan')
    hist, _ = np.histogram(pvals, bins=bins, range=(0,1), density=True)
    ece = float(np.sum(np.abs(hist - 1.0)) * (1.0 / bins))
    return ece
