from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def tau_cp(alpha: float, K: int = 2) -> float:
    return float(1.0 + (K - 1) * alpha)


def compute_bounds(alpha_list) -> pd.DataFrame:
    rows = []
    for a in alpha_list:
        tau = tau_cp(a, 2)
        prod = tau * tau
        rows.append({'alpha': float(a), 'tau': tau, 'prod_tau2': prod})
    return pd.DataFrame(rows)


def plot_bounds(df: pd.DataFrame, out_png: str):
    plt.figure(figsize=(5,4))
    plt.plot(df['alpha'], df['prod_tau2'], marker='o', label='(1+alpha)^2 (CP expected)')
    plt.xlabel('alpha'); plt.ylabel('Expected |C_X||C_Z|'); plt.title('Analytic CP Product Bound')
    plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
