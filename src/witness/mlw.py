from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from ..witness.cmi import tv_distance_to_uniform
from ..sims.lhv_detection_loophole import simulate_lhv_garg_mermin
from ..sims.lhv_memory_loophole import simulate_lhv_memory_loophole
from ..sims.lhv_communication_loophole import simulate_lhv_communication
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def pvals_binary_label_conditional(scores_cal, y_cal, scores_test):
    """
    scores_* are P(y=+1 | x). Return p_pos, p_neg for each test point
    using label-conditional split-CP (no smoothing).
    """
    scores_cal = np.asarray(scores_cal, float)
    y_cal = np.asarray(y_cal, int)
    scores_test = np.asarray(scores_test, float)

    nonc_pos_cal = 1.0 - scores_cal[y_cal == 1]
    nonc_neg_cal = scores_cal[y_cal == 0]
    n_pos = len(nonc_pos_cal)
    n_neg = len(nonc_neg_cal)
    nonc_pos_sorted = np.sort(nonc_pos_cal)
    nonc_neg_sorted = np.sort(nonc_neg_cal)

    def emp_p(nonc_sorted, v, n):
        k = n - np.searchsorted(nonc_sorted, v, side='left')
        return (k + 1) / (n + 1) if n > 0 else 1.0

    nonc_pos_test = 1.0 - scores_test
    nonc_neg_test = scores_test
    p_pos = np.array([emp_p(nonc_pos_sorted, v, n_pos) for v in nonc_pos_test], dtype=float)
    p_neg = np.array([emp_p(nonc_neg_sorted, v, n_neg) for v in nonc_neg_test], dtype=float)
    return p_pos, p_neg


def build_features(sim, cp_result, alpha: float) -> Dict[str, float]:
    s = float(sim['s_value'])
    avgX = cp_result['avg_card_by_ctx'].get(0, 2.0)
    avgZ = cp_result['avg_card_by_ctx'].get(1, 2.0)
    prod = avgX * avgZ
    p0 = cp_result['pvals_by_ctx'].get(0, np.array([]))
    p1 = cp_result['pvals_by_ctx'].get(1, np.array([]))
    tv0 = tv_distance_to_uniform(p0)
    tv1 = tv_distance_to_uniform(p1)
    covX = cp_result['coverage_by_ctx'].get(0, np.nan)
    covZ = cp_result['coverage_by_ctx'].get(1, np.nan)
    # click rates
    cA = float(np.mean(sim['meta']['clickA'])) if 'meta' in sim else np.nan
    cB = float(np.mean(sim['meta']['clickB'])) if 'meta' in sim else np.nan
    return {
        'avgX': avgX, 'avgZ': avgZ, 'prod': prod,
        'tv0': tv0, 'tv1': tv1, 'covX': covX, 'covZ': covZ,
        'clickA': cA, 'clickB': cB,
    }


def simulate_dataset(sim_fn, angles, grid, rep: int, alpha: float, method: str, K: int, h: float, calib: str, rng: np.random.Generator,
                    include_colored: bool = True, include_jitter_bias: bool = True):
    from ..cp.split import split_conformal_binary
    rows = []
    labels = []
    for gp in grid:
        p, eta = gp['p_depol'], gp['etaA']
        for r in range(rep):
            noise = {'p_depol': p}
            if include_colored and (r % 2 == 1):
                noise.update({'px': 0.02, 'pz': 0.05})
            if include_jitter_bias and (r % 3 == 2):
                noise.update({'jitter_sigma': 0.02, 'bias_delta': 0.01})
            sim = sim_fn(4000, angles, noise, {'etaA': eta, 'etaB': eta}, seed=int(rng.integers(0, 9_999_999)))
            cp = split_conformal_binary(sim['a'], sim['b'], sim['x'], sim['y'], sim['meta']['clickA'], sim['meta']['clickB'], alpha,
                                        method=method, K=K, h=h, calibration=calib)
            feat = build_features(sim, cp, alpha)
            rows.append(feat)
            labels.append(1 if sim['s_value'] > 2.1 else 0)
    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y


def train_mlw(X_train: pd.DataFrame, y_train: np.ndarray) -> GradientBoostingClassifier:
    gb = GradientBoostingClassifier(random_state=0)
    gb.fit(X_train.values, y_train)
    return gb


def conformalize_scores(scores_cal: np.ndarray, y_cal: np.ndarray, scores_test: np.ndarray, alpha: float) -> Tuple[float, np.ndarray]:
    """Split-conformal for binary: use scores near 1 as positive evidence.
    Nonconformity = 1 - score. Return threshold t such that p>=t enters set.
    """
    s = scores_cal
    nonc = 1 - s
    nonc_sorted = np.sort(nonc)
    n = len(nonc_sorted)
    k = int(np.ceil((1 - alpha) * (n + 1))) - 1
    k = np.clip(k, 0, n - 1)
    t = 1 - nonc_sorted[k]
    return float(t), (scores_test >= t).astype(int)


def evaluate_mlw(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
    scores = model.predict_proba(X_test.values)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, scores)
    pr_auc = auc(rec, prec)
    return {'roc_auc': float(roc_auc), 'pr_auc': float(pr_auc)}


def grid_search_models(X: pd.DataFrame, y: np.ndarray, cv_splits: int = 5) -> Tuple[str, dict]:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=0)
    models = {
        'logreg': (LogisticRegression(max_iter=2000, solver='saga'), {'C': [0.1, 1, 10]}),
        'svm_rbf': (SVC(probability=True), {'C': [0.5, 1, 2], 'gamma': ['scale', 0.1, 0.01]}),
        'rf': (RandomForestClassifier(random_state=0), {'n_estimators': [200, 400], 'max_depth': [None, 6, 10]}),
        'gbm': (GradientBoostingClassifier(random_state=0), {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
        'knn': (KNeighborsClassifier(), {'n_neighbors': [5, 11, 21]}),
        'gnb': (GaussianNB(), {}),
    }
    try:
        from xgboost import XGBClassifier
        models['xgb'] = (XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_estimators=300), {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]})
    except Exception:
        pass
    best = ('', -np.inf, None, None)
    report = {}
    for name, (clf, grid) in models.items():
        gs = GridSearchCV(clf, grid, cv=cv, scoring='roc_auc', n_jobs=1)
        gs.fit(X.values, y)
        score = float(gs.best_score_)
        report[name] = {'best_params': gs.best_params_, 'best_score': score}
        if score > best[1]:
            best = (name, score, gs.best_estimator_, gs.best_params_)
    return best[0], {'best_model': best[0], 'best_score': best[1], 'best_params': best[3], 'report': report}


def leakage_report(X: pd.DataFrame, y: np.ndarray) -> dict:
    rep = {}
    rep['n_rows'] = int(len(X))
    rep['n_dups'] = int(len(X) - len(X.drop_duplicates()))
    # correlations with label
    corrs = {}
    yv = y.astype(float)
    for col in X.columns:
        try:
            c = float(np.corrcoef(X[col].values.astype(float), yv)[0,1])
            corrs[col] = c
        except Exception:
            continue
    rep['high_corr_with_label'] = {k: v for k, v in corrs.items() if abs(v) > 0.98}
    rep['near_constant'] = [c for c in X.columns if np.std(X[c].values) < 1e-6]
    return rep


def kmeans_elbow(X: pd.DataFrame, kmin: int = 2, kmax: int = 10) -> dict:
    inertia = []
    sil = []
    Xv = X.values
    for k in range(kmin, kmax+1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(Xv)
        inertia.append(float(km.inertia_))
        if k > 1:
            try:
                sil.append(float(silhouette_score(Xv, labels)))
            except Exception:
                sil.append(float('nan'))
        else:
            sil.append(float('nan'))
    return {'k': list(range(kmin, kmax+1)), 'inertia': inertia, 'silhouette': sil}


def pca_2d(X: pd.DataFrame) -> dict:
    p = PCA(n_components=2, random_state=0)
    Z = p.fit_transform(X.values)
    return {'Z': Z, 'explained_var': p.explained_variance_ratio_.tolist()}


def plot_elbow(elbow: dict, out_png: str):
    plt.figure(figsize=(5,4))
    ks = elbow['k']; inertia = elbow['inertia']
    plt.plot(ks, inertia, marker='o')
    plt.xlabel('k'); plt.ylabel('Inertia'); plt.title('KMeans Elbow')
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def plot_pca(Z: np.ndarray, y: np.ndarray, out_png: str, title: str = 'PCA 2D'):
    plt.figure(figsize=(5,4))
    sns.scatterplot(x=Z[:,0], y=Z[:,1], hue=y, palette='coolwarm', s=12, linewidth=0)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def conformal_curve(scores_cal: np.ndarray, y_cal: np.ndarray, scores_test: np.ndarray, y_test: np.ndarray,
                    alphas: np.ndarray) -> pd.DataFrame:
    rows = []
    p_pos, p_neg = pvals_binary_label_conditional(scores_cal, y_cal, scores_test)
    for a in alphas:
        in_pos = (p_pos > float(a))
        in_neg = (p_neg > float(a))
        covered = np.where(y_test == 1, in_pos, in_neg)
        cov = float(np.mean(covered))
        rows.append({'alpha': float(a), 'coverage': cov})
    return pd.DataFrame(rows)


def simulate_lhv(N: int, angles: Tuple[float, float, float, float], noise_cfg: Dict, detect_cfg: Dict, seed: int | None = None) -> Dict:
    """Simulate a Local Hidden Variable model (LHV) that mimics CHSH setup but is classical.
    Strategy: Deterministic hidden variable lambda ~ Uniform(0, 2pi).
    A(a) = sign(cos(a - lambda)), B(b) = -sign(cos(b - lambda)).
    This satisfies Bell inequalities (S <= 2).
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 2*np.pi, size=N)
    A0, A1, B0, B1 = angles
    
    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    
    theta_a = np.where(a == 0, A0, A1)
    theta_b = np.where(b == 0, B0, B1)
    
    x_raw = np.sign(np.cos(theta_a - lam))
    y_raw = -np.sign(np.cos(theta_b - lam))
    
    # Fix zeros
    x_raw[x_raw == 0] = 1
    y_raw[y_raw == 0] = -1
    
    x = x_raw.astype(int)
    y = y_raw.astype(int)
    
    etaA = float(detect_cfg.get('etaA', 1.0))
    etaB = float(detect_cfg.get('etaB', 1.0))
    
    clickA = rng.random(N) < etaA
    clickB = rng.random(N) < etaB
    
    from ..sims.chsh import estimate_chsh_S
    S = estimate_chsh_S(x, y, a, b)
    
    return {
        'x': x, 'y': y, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'clickA': clickA,
            'clickB': clickB,
            'etaA': etaA, 'etaB': etaB, 'darkA': 0.0, 'darkB': 0.0,
        }
    }


def generate_adversarial_dataset(angles, grid, rep: int, alpha: float, method: str, K: int, h: float, calib: str, rng: np.random.Generator):
    """Generate a dataset containing Quantum, Simple LHV, and Hard LHV (Garg-Mermin) examples."""
    from ..sims.chsh import simulate_chsh
    
    # 1. Quantum with Colored Noise (Label 1)
    X_q, y_q = simulate_dataset(simulate_chsh, angles, grid, rep=rep, alpha=alpha, method=method, K=K, h=h, calib=calib, rng=rng,
                                include_colored=True, include_jitter_bias=True)
    
    # 2. Simple LHV (Label 0)
    X_lhv1, _ = simulate_dataset(simulate_lhv, angles, grid, rep=rep, alpha=alpha, method=method, K=K, h=h, calib=calib, rng=rng,
                                 include_colored=False, include_jitter_bias=False)
    y_lhv1 = np.zeros(len(X_lhv1), dtype=int)
    
    # 3. Hard LHV (Garg-Mermin) (Label 0)
    X_lhv2, _ = simulate_dataset(simulate_lhv_garg_mermin, angles, grid, rep=rep, alpha=alpha, method=method, K=K, h=h, calib=calib, rng=rng,
                                 include_colored=False, include_jitter_bias=False)
    y_lhv2 = np.zeros(len(X_lhv2), dtype=int)

    # 4. Memory Loophole LHV (Label 0)
    X_lhv3, _ = simulate_dataset(simulate_lhv_memory_loophole, angles, grid, rep=rep, alpha=alpha, method=method, K=K, h=h, calib=calib, rng=rng,
                                 include_colored=False, include_jitter_bias=False)
    y_lhv3 = np.zeros(len(X_lhv3), dtype=int)
    
    # 5. Communication Loophole LHV (Label 0)
    # This model uses signaling (non-local but classical)
    X_lhv4, _ = simulate_dataset(simulate_lhv_communication, angles, grid, rep=rep, alpha=alpha, method=method, K=K, h=h, calib=calib, rng=rng,
                                 include_colored=False, include_jitter_bias=False)
    y_lhv4 = np.zeros(len(X_lhv4), dtype=int)
    
    # Combine all
    X = pd.concat([X_q, X_lhv1, X_lhv2, X_lhv3, X_lhv4], ignore_index=True)
    y = np.concatenate([y_q, y_lhv1, y_lhv2, y_lhv3, y_lhv4])
    
    return X, y
