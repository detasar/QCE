from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from scipy import stats


def load_or_raise(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if 'label' not in df.columns:
        raise ValueError('Dataset must contain label column')
    return df


def fit_rf(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=400, random_state=0)
    rf.fit(X.values, y)
    return rf


def feature_importances(rf: RandomForestClassifier, X_cols: list[str]) -> pd.DataFrame:
    imp = rf.feature_importances_
    return pd.DataFrame({'feature': X_cols, 'gini_importance': imp}).sort_values('gini_importance', ascending=False)


def perm_importances(rf: RandomForestClassifier, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    r = permutation_importance(rf, X.values, y, scoring='roc_auc', n_repeats=5, random_state=0)
    return pd.DataFrame({'feature': X.columns, 'perm_importance': r.importances_mean}).sort_values('perm_importance', ascending=False)


def ks_by_feature(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        xb = X[c].values
        d0 = xb[y==0]; d1 = xb[y==1]
        if len(d0) > 0 and len(d1) > 0:
            ks = stats.ks_2samp(d0, d1)
            rows.append({'feature': c, 'ks_stat': float(ks.statistic), 'ks_p': float(ks.pvalue)})
    return pd.DataFrame(rows).sort_values('ks_stat', ascending=False)


def plot_importance(df: pd.DataFrame, col: str, title: str, out_png: str, topn: int = 15):
    d = df.head(topn)
    plt.figure(figsize=(6,4))
    sns.barplot(x=col, y='feature', data=d, orient='h')
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def rule_baseline_metrics(X: pd.DataFrame, y: np.ndarray) -> dict:
    # simple rule: S > 2 => positive
    s = X['S'].values
    yhat = (s > 2.0).astype(int)
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    try:
        roc = roc_auc_score(y, yhat)
    except Exception:
        roc = float('nan')
    prec, rec, _ = precision_recall_curve(y, yhat)
    pr = auc(rec, prec)
    acc = float(np.mean(yhat == y))
    return {'rule': 'S>2', 'roc_auc': float(roc), 'pr_auc': float(pr), 'acc': acc, 'pos_rate': float(yhat.mean())}
