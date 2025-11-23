from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Any
from .scores import (
    frequency_model,
    nonconformity_from_probs,
    knn_weighted_probs,
    fit_platt,
    apply_platt,
    fit_isotonic,
)


def split_indices(n: int, ratios=(0.6, 0.2, 0.2), rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(ratios[0] * n)
    n_cal = int(ratios[1] * n)
    tr = idx[:n_tr]
    cal = idx[n_tr:n_tr + n_cal]
    te = idx[n_tr + n_cal:]
    return tr, cal, te


def split_conformal_binary(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray,
                           clickA: np.ndarray, clickB: np.ndarray, alpha: float,
                           rng: np.random.Generator | None = None,
                           method: str = 'knn', K: int = 11, h: float = 0.25,
                           calibration: str = 'none') -> Dict[str, Any]:
    """Build CP sets for predicting Alice's outcome y_A using Bob's observed bit and b as side-info.
    Mondrian by a-context. Returns per-context average set cardinalities and p-values by context.
    """
    rng = rng or np.random.default_rng()
    N = len(y)
    # use only shots where both clicked for training/usage
    mask = clickA & clickB
    idx = np.flatnonzero(mask)
    if len(idx) < 10:
        return {'avg_card_by_ctx': {0: 2.0, 1: 2.0}, 'pvals_by_ctx': {0: np.array([]), 1: np.array([])}}
    a = a[idx]
    b = b[idx]
    ya = y[idx]
    # in simulation x=Alice, y=Bob; predict x (Alice) from (b, y_B)
    xA = x[idx]
    yB = y[idx]

    # features: vector [b, 1{yB=+1}]
    feat_vec = np.stack([b.astype(float), (yB == 1).astype(float)], axis=1)
    # also keep discrete code for frequency model
    feat_code = 2 * b + (yB == 1).astype(int)
    groups = a

    tr, cal, te = split_indices(len(idx), rng=rng)
    featv_tr, featc_tr, xA_tr, a_tr = feat_vec[tr], feat_code[tr], xA[tr], groups[tr]
    featv_cal, featc_cal, xA_cal, a_cal = feat_vec[cal], feat_code[cal], xA[cal], groups[cal]
    featv_te, featc_te, a_te = feat_vec[te], feat_code[te], groups[te]

    # split a portion of training for calibrator fitting to avoid leaking CP calibration
    if len(featv_tr) >= 10:
        n_tr = len(featv_tr)
        n_val = max(5, int(0.2 * n_tr))
        val_idx = np.arange(n_tr - n_val, n_tr)
        tr_idx_inner = np.arange(0, n_tr - n_val)
    else:
        tr_idx_inner = np.arange(0, len(featv_tr))
        val_idx = np.array([], dtype=int)

    # fit per-group frequency models
    cards_by_ctx: Dict[int, List[int]] = {0: [], 1: []}
    pvals_by_ctx: Dict[int, List[float]] = {0: [], 1: []}
    coverage_by_ctx: Dict[int, List[int]] = {0: [], 1: []}

    for ctx in (0, 1):
        tr_idx = np.flatnonzero(a_tr == ctx)
        cal_idx = np.flatnonzero(a_cal == ctx)
        te_idx = np.flatnonzero(a_te == ctx)
        if len(tr_idx) == 0 or len(cal_idx) == 0 or len(te_idx) == 0:
            continue
        # Fit base model per context
        if method == 'freq':
            model = frequency_model(featc_tr[tr_idx_inner][a_tr[tr_idx_inner] == ctx], xA_tr[tr_idx_inner][a_tr[tr_idx_inner] == ctx])
            # define predictor function
            def predict_proba(Xq_vec, Xq_code):
                p = np.array([model.get(f, {+1: 0.5}).get(+1, 0.5) for f in Xq_code], dtype=float)
                return p
        else:
            Xtr_ctx = featv_tr[tr_idx_inner][a_tr[tr_idx_inner] == ctx]
            ytr_ctx = xA_tr[tr_idx_inner][a_tr[tr_idx_inner] == ctx]
            def predict_proba(Xq_vec, Xq_code):
                if len(Xtr_ctx) == 0:
                    return np.full(Xq_vec.shape[0], 0.5)
                return knn_weighted_probs(Xtr_ctx, ytr_ctx, Xq_vec, K=K, h=h)

        # Optional calibrator fit on train split (val subset)
        calibrator = None
        if calibration in ('platt', 'isotonic') and len(val_idx) > 0:
            Xval_ctx = featv_tr[val_idx][a_tr[val_idx] == ctx]
            yval_ctx = xA_tr[val_idx][a_tr[val_idx] == ctx]
            if len(Xval_ctx) >= 5:
                raw_p_val = predict_proba(Xval_ctx, None)
                if calibration == 'platt':
                    AB = fit_platt(raw_p_val, yval_ctx)
                    calibrator = ('platt', AB)
                else:
                    ir = fit_isotonic(raw_p_val, yval_ctx)
                    calibrator = ('isotonic', ir)

        def apply_calibration(p):
            if calibrator is None:
                return p
            kind, obj = calibrator
            if kind == 'platt':
                return apply_platt(p, obj)
            else:
                return obj.predict(p)

        # Calibration nonconformities per candidate label using CP calibration set
        cal_scores_pos: List[float] = []
        cal_scores_neg: List[float] = []
        Xcal_ctx_vec = featv_cal[cal_idx][a_cal[cal_idx] == ctx]
        Xcal_ctx_code = featc_cal[cal_idx][a_cal[cal_idx] == ctx]
        if len(Xcal_ctx_vec) > 0:
            p_raw = predict_proba(Xcal_ctx_vec, Xcal_ctx_code)
            p_cal = apply_calibration(p_raw)
            s_pos_arr = 1.0 - p_cal
            s_neg_arr = 1.0 - (1.0 - p_cal)
            cal_scores_pos = np.sort(s_pos_arr)
            cal_scores_neg = np.sort(s_neg_arr)

        # Evaluate p-values and sets on test
        Xte_ctx_vec = featv_te[te_idx][a_te[te_idx] == ctx]
        Xte_ctx_code = featc_te[te_idx][a_te[te_idx] == ctx]
        yte_ctx = xA[te][a_te == ctx]
        if len(Xte_ctx_vec) == 0:
            continue
        p_raw_te = predict_proba(Xte_ctx_vec, Xte_ctx_code)
        p_te = apply_calibration(p_raw_te)

        def pval(arr, s):
            n = len(arr)
            if n == 0:
                return 1.0
            ge = n - np.searchsorted(arr, s, side='left')
            return (ge + 1) / (n + 1)

        for i in range(len(Xte_ctx_vec)):
            s_pos = 1.0 - p_te[i]
            s_neg = 1.0 - (1.0 - p_te[i])
            p_pos_val = pval(cal_scores_pos, s_pos)
            p_neg_val = pval(cal_scores_neg, s_neg)
            pvals_by_ctx[ctx].append(p_pos_val)
            # set membership for true label
            in_set = False
            if yte_ctx[i] == 1 and p_pos_val > alpha:
                in_set = True
            if yte_ctx[i] == -1 and p_neg_val > alpha:
                in_set = True
            coverage_by_ctx[ctx].append(1 if in_set else 0)
            # cardinality
            card = (1 if p_pos_val > alpha else 0) + (1 if p_neg_val > alpha else 0)
            if card == 0:
                card = 1
            cards_by_ctx[ctx].append(card)

    avg_card = {ctx: float(np.mean(cards_by_ctx[ctx])) if cards_by_ctx[ctx] else 2.0 for ctx in (0, 1)}
    coverage = {ctx: (np.mean(coverage_by_ctx[ctx]) if coverage_by_ctx[ctx] else np.nan) for ctx in (0, 1)}
    return {
        'avg_card_by_ctx': avg_card,
        'pvals_by_ctx': {ctx: np.array(pvals_by_ctx[ctx], dtype=float) for ctx in (0, 1)},
        'coverage_by_ctx': coverage
    }
