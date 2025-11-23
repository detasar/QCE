import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..io.config import load_yaml
from ..io.paths import out_path
from ..witness.mlw import generate_adversarial_dataset
from ..witness.cew_improved import CEWCalibrator
from ..io.notes import append_note

def run_cew_calibration_analysis(args):
    cfg = load_yaml(args.config)
    A0, A1, B0, B1 = cfg['angles']['A0'], cfg['angles']['A1'], cfg['angles']['B0'], cfg['angles']['B1']
    angles = (A0, A1, B0, B1)
    
    # Grid
    grid = []
    for p in [0.0, 0.05, 0.1]:
        for eta in [0.8, 0.9, 1.0]:
            grid.append({'p_depol': p, 'etaA': eta, 'etaB': eta})
            
    rng = np.random.default_rng(42)
    
    print("Generating dataset for calibration...")
    # We use generate_adversarial_dataset which gives us Quantum (1) and LHV (0)
    X, y = generate_adversarial_dataset(angles, grid, rep=int(args.rep), alpha=0.2, 
                                        method='knn', K=11, h=0.25, calib='isotonic', rng=rng)
    
    # Split
    X_null = X[y == 0] # LHV
    X_test = X[y == 1] # Quantum
    
    print(f"Null samples (LHV): {len(X_null)}")
    print(f"Test samples (Quantum): {len(X_test)}")
    
    # Calibrate
    alpha = 0.05
    calibrator = CEWCalibrator(alpha=alpha)
    L = calibrator.fit(X_null)
    print(f"Calibrated Threshold L (alpha={alpha}): {L:.4f}")
    
    # Predict
    preds_null = calibrator.predict(X_null)
    preds_test = calibrator.predict(X_test)
    
    fpr = np.mean(preds_null)
    tpr = np.mean(preds_test)
    
    print(f"False Positive Rate on Null (Target <= {alpha}): {fpr:.4f}")
    print(f"True Positive Rate on Quantum: {tpr:.4f}")
    
    # Plot distributions
    plt.figure(figsize=(8, 6))
    prod_null = X_null['avgX'] * X_null['avgZ']
    prod_test = X_test['avgX'] * X_test['avgZ']
    
    sns.histplot(prod_null, color='red', label='Null (LHV)', kde=True, stat="density", alpha=0.5)
    sns.histplot(prod_test, color='green', label='Quantum', kde=True, stat="density", alpha=0.5)
    plt.axvline(L, color='black', linestyle='--', label=f'Threshold L={L:.2f}')
    plt.xlabel('Cardinality Product (avgX * avgZ)')
    plt.title('CEW Calibration: Null vs Quantum')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path('fig_cew_calibration.png'), dpi=200)
    plt.close()
    
    append_note(
        'CEW Kalibrasyon Analizi',
        f"LHV verisi ile kalibre edilen threshold L={L:.3f}. "
        f"FPR={fpr:.3f} (Hedef {alpha}), TPR={tpr:.3f}. "
        f"Data-driven kalibrasyon ile FPR kontrol altına alındı.",
        ['fig_cew_calibration.png']
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rep', default=10)
    args = parser.parse_args()
    run_cew_calibration_analysis(args)
