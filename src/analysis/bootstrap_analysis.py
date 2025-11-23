import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
import sys
import os

# Fix import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def bootstrap_s_value(data_csv: str, n_bootstrap: int = 1000) -> Dict:
    """
    Compute bootstrap confidence interval for CHSH S-value.
    
    Args:
        data_csv: Path to IBM data CSV (a, b, x, y format)
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with mean, std, and 95% CI bounds
    """
    from src.sims.chsh import estimate_chsh_S
    
    df = pd.read_csv(data_csv)
    n = len(df)
    
    s_values = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        
        s = estimate_chsh_S(
            sample['x'].values,
            sample['y'].values,
            sample['a'].values,
            sample['b'].values
        )
        s_values.append(s)
    
    s_values = np.array(s_values)
    
    return {
        'mean': float(np.mean(s_values)),
        'std': float(np.std(s_values)),
        'ci_lower': float(np.percentile(s_values, 2.5)),
        'ci_upper': float(np.percentile(s_values, 97.5)),
        'n_bootstrap': n_bootstrap,
        'original_s': float(estimate_chsh_S(df['x'].values, df['y'].values, df['a'].values, df['b'].values))
    }

def bootstrap_ml_auc(X: pd.DataFrame, y: np.ndarray, n_bootstrap: int = 100) -> Dict:
    """
    Compute bootstrap confidence interval for ML witness AUC.
    
    Args:
        X: Feature matrix (conformal features)
        y: Labels (0=LHV, 1=Quantum)
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with mean AUC and 95% CI
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    aucs = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_bootstrap):
        # Split and resample
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=rng.integers(0, 10000), stratify=y
        )
        
        # Train
        model = GradientBoostingClassifier(random_state=0, n_estimators=50)
        model.fit(X_train, y_train)
        
        # Predict
        scores = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, scores)
        aucs.append(auc)
    
    aucs = np.array(aucs)
    
    return {
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs)),
        'ci_lower': float(np.percentile(aucs, 2.5)),
        'ci_upper': float(np.percentile(aucs, 97.5)),
        'n_bootstrap': n_bootstrap
    }

def power_analysis(target_ci_width: float = 0.02, alpha_chsh: float = 2.828) -> Dict:
    """
    Estimate required sample size for desired confidence interval width.
    
    Args:
        target_ci_width: Desired half-width of 95% CI
        alpha_chsh: True CHSH parameter (Tsirelson bound)
        
    Returns:
        Estimated required shots
    """
    # Empirical formula from binary outcome variance
    # CI width ≈ 1.96 * sqrt(var / n)
    # For CHSH, var ≈ 0.5 (empirical from our data)
    
    var_estimate = 0.5
    z_score = 1.96  # 95% CI
    
    n_required = int((z_score * np.sqrt(var_estimate) / target_ci_width) ** 2)
    
    return {
        'required_shots': n_required,
        'target_ci_width': target_ci_width,
        'confidence_level': 0.95
    }

if __name__ == "__main__":
    print("Running Bootstrap Analysis...")
    
    # 1. Bootstrap S-value
    print("\n=== S-value Bootstrap ===")
    s_stats = bootstrap_s_value('ibm_data_1763861062.csv', n_bootstrap=1000)
    print(f"Original S: {s_stats['original_s']:.4f}")
    print(f"Bootstrap Mean: {s_stats['mean']:.4f}")
    print(f"Bootstrap Std: {s_stats['std']:.4f}")
    print(f"95% CI: [{s_stats['ci_lower']:.4f}, {s_stats['ci_upper']:.4f}]")
    
    # Save results
    with open('bootstrap_s_value.json', 'w') as f:
        json.dump(s_stats, f, indent=2)
    
    # 2. Power analysis
    print("\n=== Power Analysis ===")
    power = power_analysis(target_ci_width=0.01)
    print(f"For CI width ±{power['target_ci_width']}: need {power['required_shots']} shots")
    
    with open('power_analysis.json', 'w') as f:
        json.dump(power, f, indent=2)
    
    print("\n✅ Bootstrap analysis complete!")
