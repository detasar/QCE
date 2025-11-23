import numpy as np
import pandas as pd
from pathlib import Path
import sys
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.sims.chsh import simulate_chsh
from src.sims.lhv_detection_loophole import simulate_lhv_garg_mermin
from src.sims.lhv_memory_loophole import simulate_lhv_memory_loophole
from src.sims.lhv_communication_loophole import simulate_lhv_communication
from src.cp.split import split_indices

# Configuration
OUTPUT_DIR = Path("results/refinement")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = Path("figures/refinement")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.1
N_SHOTS = 1000
N_DATASETS = 200  # Number of datasets per regime
SEED = 42

def generate_dataset(kind='quantum', params=None, seed=None):
    """Generate a single dataset."""
    if params is None:
        params = {}
    
    # Default angles (optimal CHSH)
    angles = (0, -np.pi/2, -np.pi/4, np.pi/4)
    
    noise_cfg = {}
    detect_cfg = {}
    
    if kind == 'quantum':
        noise_cfg['p_depol'] = params.get('p_depol', 0.0)
        detect_cfg['etaA'] = params.get('eta', 1.0)
        detect_cfg['etaB'] = params.get('eta', 1.0)
        return simulate_chsh(N_SHOTS, angles, noise_cfg, detect_cfg, seed=seed)
    elif kind == 'lhv_detection':
        detect_cfg['etaA'] = params.get('eta', 0.85) # Example low efficiency
        detect_cfg['etaB'] = params.get('eta', 0.85)
        return simulate_lhv_garg_mermin(N_SHOTS, angles, noise_cfg, detect_cfg, seed=seed)
    elif kind == 'lhv_memory':
        noise_cfg['mu'] = params.get('mu', 0.2)
        return simulate_lhv_memory_loophole(N_SHOTS, angles, noise_cfg, detect_cfg, seed=seed)
    elif kind == 'lhv_comm':
        noise_cfg['epsilon'] = params.get('epsilon', 0.05)
        return simulate_lhv_communication(N_SHOTS, angles, noise_cfg, detect_cfg, seed=seed)
    else:
        raise ValueError(f"Unknown kind: {kind}")

def extract_features(sim):
    """Extract features (X), labels (y), and contexts (C)."""
    mask = sim['meta']['clickA'] & sim['meta']['clickB']
    a = sim['a'][mask]
    b = sim['b'][mask]
    x = sim['x'][mask] # Label
    y = sim['y'][mask] # Feature
    
    # Features: [b, y]
    feat_vec = np.stack([b.astype(float), (y == 1).astype(float)], axis=1)
    
    return feat_vec, x, a

def train_calibrated_predictor(X_tr, y_tr, C_tr, X_cal, y_cal, C_cal):
    """Train RF and compute calibration scores per context."""
    models = {}
    cal_scores = {0: [], 1: []}
    
    for ctx in [0, 1]:
        # Train
        tr_mask = (C_tr == ctx)
        if tr_mask.sum() < 10:
            models[ctx] = None
            continue
            
        clf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1)
        clf.fit(X_tr[tr_mask], y_tr[tr_mask])
        models[ctx] = clf
        
        # Calibrate
        cal_mask = (C_cal == ctx)
        if cal_mask.sum() < 10:
            continue
            
        probs = clf.predict_proba(X_cal[cal_mask])
        class_map = {c: i for i, c in enumerate(clf.classes_)}
        
        y_true = y_cal[cal_mask]
        p_true = np.array([probs[i, class_map[yi]] for i, yi in enumerate(y_true)])
        
        scores = 1.0 - p_true
        cal_scores[ctx] = np.sort(scores)
        
    return models, cal_scores

def evaluate_predictor(models, cal_scores, X_te, y_te, C_te):
    """Evaluate p-values, coverage, and CMI."""
    n_te = len(y_te)
    p_vals_out = np.zeros(n_te)
    cov_out = np.zeros(n_te)
    card_out = np.zeros(n_te)
    
    def get_pval(scores, s):
        if len(scores) == 0: return 1.0
        idx = np.searchsorted(scores, s, side='left')
        ge = len(scores) - idx
        return (ge + 1) / (len(scores) + 1)

    for ctx in [0, 1]:
        mask = (C_te == ctx)
        if mask.sum() == 0: continue
        
        clf = models.get(ctx)
        scores_cal = cal_scores.get(ctx, [])
        
        if clf is None or len(scores_cal) == 0:
            p_vals_out[mask] = 1.0
            cov_out[mask] = 1.0
            card_out[mask] = 2.0
            continue
            
        X_ctx = X_te[mask]
        y_ctx = y_te[mask]
        
        probs = clf.predict_proba(X_ctx)
        class_map = {c: i for i, c in enumerate(clf.classes_)}
        
        if 1 in class_map:
            p_hat_1 = probs[:, class_map[1]]
        else:
            p_hat_1 = np.zeros(len(X_ctx))
        s_1 = 1.0 - p_hat_1
        pval_1 = np.array([get_pval(scores_cal, s) for s in s_1])
        
        if -1 in class_map:
            p_hat_m1 = probs[:, class_map[-1]]
        else:
            p_hat_m1 = np.zeros(len(X_ctx))
        s_m1 = 1.0 - p_hat_m1
        pval_m1 = np.array([get_pval(scores_cal, s) for s in s_m1])
        
        in_set_1 = (pval_1 > ALPHA)
        in_set_m1 = (pval_m1 > ALPHA)
        
        batch_p_true = []
        batch_cov = []
        batch_card = []
        
        for i, yi in enumerate(y_ctx):
            if yi == 1:
                p_true = pval_1[i]
                is_covered = in_set_1[i]
            else:
                p_true = pval_m1[i]
                is_covered = in_set_m1[i]
                
            batch_p_true.append(p_true)
            batch_cov.append(is_covered)
            batch_card.append(int(in_set_1[i]) + int(in_set_m1[i]))
            
        p_vals_out[mask] = np.array(batch_p_true)
        cov_out[mask] = np.array(batch_cov)
        card_out[mask] = np.array(batch_card)
        
    B = 20
    bins = np.linspace(0, 1, B+1)
    counts, _ = np.histogram(p_vals_out, bins=bins)
    cdf = np.cumsum(counts) / n_te
    ideal_cdf = bins[1:]
    cmi = np.mean(np.abs(cdf - ideal_cdf))
    
    return {
        'coverage': np.mean(cov_out),
        'cmi': cmi,
        'avg_card': np.mean(card_out),
        'p_values': p_vals_out
    }

def run_r1_experiment():
    print("Running R1: Wrapper vs Witness Mode...")
    results = []
    
    # 1. Wrapper Mode
    print("  Wrapper Mode...")
    print("    Processing Wrapper LHV...")
    for i in range(N_DATASETS // 2):
        if i % 10 == 0: print(f"      Iter {i}/{N_DATASETS//2}")
        sim = generate_dataset('lhv_detection', seed=SEED+i)
        X, y, C = extract_features(sim)
        tr, cal, te = split_indices(len(y), ratios=(0.6, 0.2, 0.2))
        
        models, cal_scores = train_calibrated_predictor(X[tr], y[tr], C[tr], X[cal], y[cal], C[cal])
        metrics = evaluate_predictor(models, cal_scores, X[te], y[te], C[te])
        
        results.append({
            'mode': 'Wrapper', 'dist': 'LHV', 
            'coverage': metrics['coverage'], 'cmi': metrics['cmi']
        })
        
    print("    Processing Wrapper Quantum...")
    for i in range(N_DATASETS // 2):
        if i % 10 == 0: print(f"      Iter {i}/{N_DATASETS//2}")
        sim = generate_dataset('quantum', seed=SEED+1000+i)
        X, y, C = extract_features(sim)
        tr, cal, te = split_indices(len(y), ratios=(0.6, 0.2, 0.2))
        
        models, cal_scores = train_calibrated_predictor(X[tr], y[tr], C[tr], X[cal], y[cal], C[cal])
        metrics = evaluate_predictor(models, cal_scores, X[te], y[te], C[te])
        
        results.append({
            'mode': 'Wrapper', 'dist': 'Quantum', 
            'coverage': metrics['coverage'], 'cmi': metrics['cmi']
        })
        
    # 2. Witness Mode
    print("  Witness Mode...")
    X_pool, y_pool, C_pool = [], [], []
    for i in range(50): 
        sim = generate_dataset('lhv_detection', seed=SEED+2000+i)
        Xi, yi, Ci = extract_features(sim)
        X_pool.append(Xi); y_pool.append(yi); C_pool.append(Ci)
    
    X_pool = np.concatenate(X_pool)
    y_pool = np.concatenate(y_pool)
    C_pool = np.concatenate(C_pool)
    
    tr, cal, _ = split_indices(len(y_pool), ratios=(0.7, 0.3, 0.0))
    global_models, global_cal_scores = train_calibrated_predictor(
        X_pool[tr], y_pool[tr], C_pool[tr], 
        X_pool[cal], y_pool[cal], C_pool[cal]
    )
    
    print("    Processing Witness LHV...")
    for i in range(N_DATASETS // 2):
        if i % 10 == 0: print(f"      Iter {i}/{N_DATASETS//2}")
        sim = generate_dataset('lhv_detection', seed=SEED+3000+i)
        X, y, C = extract_features(sim)
        metrics = evaluate_predictor(global_models, global_cal_scores, X, y, C)
        results.append({
            'mode': 'Witness', 'dist': 'LHV', 
            'coverage': metrics['coverage'], 'cmi': metrics['cmi']
        })
        
    print("    Processing Witness Quantum...")
    for i in range(N_DATASETS // 2):
        if i % 10 == 0: print(f"      Iter {i}/{N_DATASETS//2}")
        sim = generate_dataset('quantum', seed=SEED+4000+i)
        X, y, C = extract_features(sim)
        metrics = evaluate_predictor(global_models, global_cal_scores, X, y, C)
        results.append({
            'mode': 'Witness', 'dist': 'Quantum', 
            'coverage': metrics['coverage'], 'cmi': metrics['cmi']
        })
        
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "r1_results.csv", index=False)
    
    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x='mode', y='cmi', hue='dist', split=True)
    plt.title("CMI Distribution: Wrapper vs Witness Mode")
    plt.savefig(FIGURES_DIR / "r1_cmi_violin.png")
    plt.close()
    
    print("R1 Complete.")

def run_r2_experiment():
    print("Running R2: CEW Baseline Sanity Check...")
    results = []
    
    Ns = [250, 500, 1000, 2000]
    R = 100 
    
    for N in Ns:
        print(f"  Processing N={N}...")
        for r in range(R):
            angles = (0, -np.pi/2, -np.pi/4, np.pi/4)
            sim = simulate_lhv_garg_mermin(N, angles, {}, {'etaA':0.85, 'etaB':0.85}, seed=SEED+r*N)
            
            X, y, C = extract_features(sim)
            if len(y) < 20: continue
            
            tr, cal, te = split_indices(len(y), ratios=(0.6, 0.2, 0.2))
            if len(te) == 0: continue
            
            models, cal_scores = train_calibrated_predictor(X[tr], y[tr], C[tr], X[cal], y[cal], C[cal])
            
            card_X = []
            card_Z = []
            
            for ctx in [0, 1]:
                mask = (C[te] == ctx)
                if mask.sum() == 0: continue
                
                clf = models.get(ctx)
                scores_cal = cal_scores.get(ctx, [])
                if clf is None or len(scores_cal) == 0: continue
                
                X_ctx = X[te][mask]
                probs = clf.predict_proba(X_ctx)
                class_map = {c: i for i, c in enumerate(clf.classes_)}
                
                if 1 in class_map: p1 = probs[:, class_map[1]]
                else: p1 = np.zeros(len(X_ctx))
                if -1 in class_map: pm1 = probs[:, class_map[-1]]
                else: pm1 = np.zeros(len(X_ctx))
                
                def get_pval(scores, s):
                    idx = np.searchsorted(scores, s, side='left')
                    return (len(scores) - idx + 1) / (len(scores) + 1)
                
                s1 = 1.0 - p1
                sm1 = 1.0 - pm1
                
                pv1 = np.array([get_pval(scores_cal, s) for s in s1])
                pvm1 = np.array([get_pval(scores_cal, s) for s in sm1])
                
                cards = (pv1 > ALPHA).astype(int) + (pvm1 > ALPHA).astype(int)
                cards[cards == 0] = 1
                
                if ctx == 0: card_X.extend(cards)
                else: card_Z.extend(cards)
            
            if len(card_X) > 0 and len(card_Z) > 0:
                mean_X = np.mean(card_X)
                mean_Z = np.mean(card_Z)
                prod = mean_X * mean_Z
                
                results.append({
                    'N': N, 'prod': prod, 'baseline': (1+ALPHA)**2
                })
                
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "r2_results.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='N', y='prod', label='Empirical Product')
    plt.axhline((1+ALPHA)**2, color='r', linestyle='--', label='(1+alpha)^2 Baseline')
    plt.title("CEW Baseline Convergence (LHV Null)")
    plt.ylabel("E[|Cx|] * E[|Cz|]")
    plt.legend()
    plt.savefig(FIGURES_DIR / "r2_convergence.png")
    plt.close()
    
    print("R2 Complete.")

if __name__ == "__main__":
    run_r1_experiment()
    run_r2_experiment()
