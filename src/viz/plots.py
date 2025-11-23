from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap(df: pd.DataFrame, x: str, y: str, z: str, cmap: str = 'viridis', levels=None, cbar=True, ax=None):
    ax = ax or plt.gca()
    pv = df.pivot(index=y, columns=x, values=z)
    sns.heatmap(pv.sort_index(ascending=True), cmap=cmap, cbar=cbar, ax=ax)
    ax.invert_yaxis()
    return ax


def phase_diagram(df: pd.DataFrame, x: str, y: str, z: str, title: str = '', out_png: str | None = None):
    plt.figure(figsize=(6, 4))
    ax = heatmap(df, x, y, z)
    ax.set_title(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        plt.close()
    else:
        return ax


def heatmap_with_hatch(df: pd.DataFrame, x: str, y: str, z: str, mask_col: str, title: str, out_png: str):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    pv = df.pivot(index=y, columns=x, values=z).sort_index(ascending=True)
    maskpv = df.pivot(index=y, columns=x, values=mask_col).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pv, cmap='viridis', cbar=True, ax=ax)
    ax.invert_yaxis()
    # Overlay hatch where mask==1
    for (i, j), val in np.ndenumerate(maskpv.values):
        if val and not np.isnan(val):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, hatch='///', edgecolor='white', linewidth=0.0))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
