import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..io.config import load_yaml
from ..io.paths import out_path, ensure_parent
from ..witness.mlw import generate_adversarial_dataset, train_mlw, evaluate_mlw
from ..io.notes import append_note

def run_adversarial_analysis(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    
    # Define a small grid for speed
    grid = []
    for p in [0.0, 0.05, 0.1]:
        for eta in [0.8, 0.9, 1.0]:
            grid.append({'p_depol': p, 'etaA': eta, 'etaB': eta})
            
    rng = np.random.default_rng(42)
    
    print("Generating adversarial dataset...")
    X, y = generate_adversarial_dataset(angles, grid, rep=int(args.rep), alpha=0.2, 
                                        method='knn', K=11, h=0.25, calib='isotonic', rng=rng)
    
    # Add a 'type' column to X for analysis (Quantum vs LHV)
    # We know the first half is Quantum, second half is LHV (based on implementation)
    n_q = len(X) // 2
    types = ['Quantum'] * n_q + ['LHV'] * (len(X) - n_q)
    X['type'] = types
    
    # Split into Train/Test
    # We want to train mostly on Quantum (standard) to see if it generalizes/fails on LHV
    # But `generate_adversarial_dataset` mixes them. 
    # Let's manually split: Train on Quantum, Test on Quantum + LHV
    
    X_q = X.iloc[:n_q]
    y_q = y[:n_q]
    X_lhv = X.iloc[n_q:]
    y_lhv = y[n_q:]
    
    # Train on 80% of Quantum
    n_tr = int(0.8 * n_q)
    indices = np.arange(n_q)
    rng.shuffle(indices)
    tr_idx = indices[:n_tr]
    te_idx = indices[n_tr:]
    
    X_train = X_q.iloc[tr_idx].drop(columns=['type'])
    y_train = y_q[tr_idx]
    
    X_test_q = X_q.iloc[te_idx].drop(columns=['type'])
    y_test_q = y_q[te_idx]
    
    X_test_lhv = X_lhv.drop(columns=['type'])
    y_test_lhv = y_lhv # Should be all 0s
    
    print(f"Training model on {len(X_train)} quantum samples...")
    model = train_mlw(X_train, y_train)
    
    # Evaluate
    print("Evaluating on Quantum Test Set...")
    metrics_q = evaluate_mlw(model, X_test_q, y_test_q)
    print(f"Quantum Test: ROC AUC = {metrics_q['roc_auc']:.4f}")
    
    print("Evaluating on LHV (Adversarial) Set...")
    # For LHV, y is all 0. ROC AUC might be undefined or 0.5 if it predicts all 0.
    # We care about False Positive Rate: How many LHV are predicted as Entangled?
    scores_lhv = model.predict_proba(X_test_lhv.values)[:, 1]
    # If model is robust, scores_lhv should be low.
    mean_score_lhv = np.mean(scores_lhv)
    fp_rate_lhv = np.mean(scores_lhv > 0.5)
    print(f"LHV Test: Mean Score = {mean_score_lhv:.4f}, FP Rate (score>0.5) = {fp_rate_lhv:.4f}")
    
    # Plot distributions
    scores_q = model.predict_proba(X_test_q.values)[:, 1]
    
    plt.figure(figsize=(8, 6))
    sns.histplot(scores_q[y_test_q==1], color='green', label='Quantum (Entangled)', kde=True, stat="density", alpha=0.5)
    sns.histplot(scores_q[y_test_q==0], color='blue', label='Quantum (Separable)', kde=True, stat="density", alpha=0.5)
    sns.histplot(scores_lhv, color='red', label='LHV (Classical)', kde=True, stat="density", alpha=0.5)
    plt.xlabel('ML Model Score (Probability of Entanglement)')
    plt.title('Adversarial Robustness: Score Distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path('fig_adversarial_scores.png'), dpi=200)
    plt.close()
    
    # Save report
    report = {
        'quantum_roc_auc': metrics_q['roc_auc'],
        'lhv_mean_score': float(mean_score_lhv),
        'lhv_fp_rate': float(fp_rate_lhv),
        'n_train': len(X_train),
        'n_test_q': len(X_test_q),
        'n_test_lhv': len(X_test_lhv)
    }
    pd.DataFrame([report]).to_csv(out_path('adversarial_report.csv'), index=False)
    
    append_note(
        'Adversarial ML Analizi',
        f"Model Quantum verisinde başarılı (AUC={metrics_q['roc_auc']:.3f}), LHV verisinde FP oranı={fp_rate_lhv:.3f}. "
        f"Ortalama LHV skoru: {mean_score_lhv:.3f}.",
        ['fig_adversarial_scores.png', 'adversarial_report.csv']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rep', default=5)
    args = parser.parse_args()
    run_adversarial_analysis(args)
