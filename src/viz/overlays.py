from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .plots import heatmap


def cew_vs_S_overlay(cew_df: pd.DataFrame, S_df: pd.DataFrame, x: str, y: str, out_png: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    heatmap(cew_df, x, y, 'CEW_rate', ax=axes[0])
    axes[0].set_title('CEW rate')
    heatmap(S_df, x, y, 'S_mean', ax=axes[1])
    axes[1].set_title('|S| mean')
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
