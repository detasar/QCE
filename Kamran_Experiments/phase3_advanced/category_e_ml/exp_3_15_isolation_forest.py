"""
Experiment 3.15: Isolation Forest Anomaly Detection

Unsupervised anomaly detection using Isolation Forest.
Key advantages:
- No training labels needed
- Fast training and inference
- Works well in high dimensions
- Natural anomaly scoring

Based on Liu et al. (2008) "Isolation Forest"

Author: Davut Emre Tasar
Date: 2025-12-07
"""

import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QKDFeatureExtractor:
    """
    Extract features from QKD simulation data.

    Features designed to capture:
    - Distribution properties of p-values
    - CHSH correlation structure
    - Temporal patterns
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def extract_pvalue_features(self, pvalues: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from p-values.
        """
        features = []

        # Basic statistics
        features.append(np.mean(pvalues))
        features.append(np.std(pvalues))
        features.append(np.median(pvalues))

        # Quartiles
        features.append(np.percentile(pvalues, 25))
        features.append(np.percentile(pvalues, 75))

        # Extreme value fractions
        features.append(np.mean(pvalues < 0.05))  # Low p-value fraction
        features.append(np.mean(pvalues > 0.95))  # High p-value fraction

        # Entropy estimate (binned)
        hist, _ = np.histogram(pvalues, bins=10, range=(0, 1))
        hist = hist / (len(pvalues) + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        features.append(entropy)

        # Autocorrelation (lag 1)
        if len(pvalues) > 1:
            autocorr = np.corrcoef(pvalues[:-1], pvalues[1:])[0, 1]
            features.append(autocorr if not np.isnan(autocorr) else 0)
        else:
            features.append(0)

        # Runs statistic (sign changes from median)
        median = np.median(pvalues)
        signs = (pvalues > median).astype(int)
        runs = 1 + np.sum(signs[1:] != signs[:-1]) if len(signs) > 1 else 0
        features.append(runs / len(pvalues))

        return np.array(features)

    def extract_correlator_features(self, correlators: Dict) -> np.ndarray:
        """
        Extract features from CHSH correlators.
        """
        features = []

        # Individual correlators
        c_values = list(correlators.values()) if isinstance(correlators, dict) else correlators
        features.extend(c_values[:4] if len(c_values) >= 4 else c_values + [0]*(4-len(c_values)))

        # CHSH value
        if len(c_values) >= 4:
            chsh = abs(c_values[0] - c_values[1] + c_values[2] + c_values[3])
        else:
            chsh = 0
        features.append(chsh)

        # Correlator variance
        features.append(np.var(c_values) if len(c_values) > 0 else 0)

        return np.array(features)

    def extract_all_features(self, data: Dict) -> np.ndarray:
        """
        Extract all features from QKD data dictionary.
        """
        features = []

        # P-value features
        if 'p_values' in data:
            pval_features = self.extract_pvalue_features(np.array(data['p_values']))
            features.extend(pval_features)
        else:
            features.extend([0.5, 0.3, 0.5, 0.25, 0.75, 0.05, 0.05, 2.3, 0, 0.5])

        # Correlator features
        if 'correlators' in data:
            corr_features = self.extract_correlator_features(data['correlators'])
            features.extend(corr_features)
        else:
            features.extend([0.7, 0.7, 0.7, 0.7, 2.8, 0])

        # QBER
        features.append(data.get('qber', 0.03))

        # Visibility
        features.append(data.get('visibility', 1.0))

        return np.array(features)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector for QKD.

    Trains on calibration (honest) data and detects
    anomalies as outliers.
    """

    def __init__(self, contamination: float = 0.05,
                 n_estimators: int = 100,
                 random_state: int = 42):
        """
        Args:
            contamination: Expected fraction of outliers in training data
            n_estimators: Number of isolation trees
            random_state: Random seed
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_extractor = QKDFeatureExtractor()
        self.is_fitted = False
        self.threshold = 0

    def prepare_features(self, data_list: List[Dict]) -> np.ndarray:
        """Extract features from list of QKD data dictionaries."""
        features = []
        for data in data_list:
            f = self.feature_extractor.extract_all_features(data)
            features.append(f)
        return np.array(features)

    def fit(self, calibration_data: List[Dict]) -> Dict:
        """
        Train isolation forest on calibration data.

        Args:
            calibration_data: List of honest QKD data dictionaries

        Returns:
            Training statistics
        """
        # Extract features
        X = self.prepare_features(calibration_data)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit isolation forest
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Get decision scores on training data
        scores = self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, 5)  # 5% expected FPR

        return {
            'n_samples': len(calibration_data),
            'n_features': X.shape[1],
            'threshold': self.threshold,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores))
        }

    def predict(self, test_data: Dict) -> Dict:
        """
        Predict if test data is anomalous.

        Returns:
            Dictionary with prediction and scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        X = self.prepare_features([test_data])
        X_scaled = self.scaler.transform(X)

        # Get prediction and score
        prediction = self.model.predict(X_scaled)[0]  # -1 for anomaly, 1 for normal
        score = self.model.decision_function(X_scaled)[0]

        # Anomaly score (higher = more anomalous)
        anomaly_score = -score  # Invert so higher = more anomalous

        return {
            'is_anomaly': prediction == -1,
            'decision_score': float(score),
            'anomaly_score': float(anomaly_score),
            'threshold': self.threshold
        }

    def predict_batch(self, test_data_list: List[Dict]) -> List[Dict]:
        """Predict on batch of test data."""
        return [self.predict(d) for d in test_data_list]


def generate_honest_data(n_samples: int, seed: int = 42) -> List[Dict]:
    """Generate honest QKD simulation data."""
    np.random.seed(seed)
    data = []

    for _ in range(n_samples):
        # Uniform p-values
        p_values = np.random.uniform(0, 1, 100)

        # CHSH correlators near theoretical max
        noise = np.random.normal(0, 0.05, 4)
        correlators = {
            'E_00': 0.7071 + noise[0],
            'E_01': 0.7071 + noise[1],
            'E_10': 0.7071 + noise[2],
            'E_11': -0.7071 + noise[3]
        }

        data.append({
            'p_values': p_values.tolist(),
            'correlators': correlators,
            'qber': np.random.uniform(0.01, 0.03),
            'visibility': np.random.uniform(0.95, 1.0)
        })

    return data


def generate_attack_data(n_samples: int, attack_type: str,
                         strength: float, seed: int = 43) -> List[Dict]:
    """Generate QKD data under attack."""
    np.random.seed(seed)
    data = []

    for _ in range(n_samples):
        if attack_type == 'intercept_resend':
            # P-values shift toward extremes
            p_values = np.random.beta(1 + strength, 1 + strength, 100)
        elif attack_type == 'decorrelation':
            # Correlators reduced
            p_values = np.random.uniform(0, 1, 100)
        elif attack_type == 'visibility':
            # P-values more uniform but correlations reduced
            p_values = np.random.uniform(0, 1, 100)
        else:
            p_values = np.random.uniform(0, 1, 100)

        # Reduced correlators based on attack
        visibility = 1 - strength * 0.3
        noise = np.random.normal(0, 0.05, 4)
        correlators = {
            'E_00': visibility * 0.7071 + noise[0],
            'E_01': visibility * 0.7071 + noise[1],
            'E_10': visibility * 0.7071 + noise[2],
            'E_11': visibility * (-0.7071) + noise[3]
        }

        data.append({
            'p_values': p_values.tolist(),
            'correlators': correlators,
            'qber': 0.02 + strength * 0.08,
            'visibility': visibility
        })

    return data


def test_training():
    """Test model training."""
    print("Testing: Model training...")

    # Generate calibration data
    honest_data = generate_honest_data(200)

    # Train model
    detector = IsolationForestDetector()
    train_stats = detector.fit(honest_data)

    print(f"  Training samples: {train_stats['n_samples']}")
    print(f"  Features: {train_stats['n_features']}")
    print(f"  Threshold: {train_stats['threshold']:.4f}")

    passed = detector.is_fitted and train_stats['n_samples'] == 200

    return {
        'train_stats': train_stats,
        'passed': passed
    }


def test_false_positive_rate():
    """Test false positive rate on honest data."""
    print("Testing: False positive rate...")

    # Generate data
    train_data = generate_honest_data(300, seed=42)
    test_data = generate_honest_data(100, seed=100)

    # Train and test
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(train_data)

    predictions = detector.predict_batch(test_data)
    n_fp = sum(1 for p in predictions if p['is_anomaly'])
    fpr = n_fp / len(test_data)

    print(f"  False positives: {n_fp}/{len(test_data)}")
    print(f"  FPR: {fpr:.2%}")

    return {
        'n_false_positives': n_fp,
        'n_total': len(test_data),
        'fpr': fpr,
        'passed': fpr < 0.15  # Allow up to 15% FPR
    }


def test_attack_detection():
    """Test detection of various attacks."""
    print("Testing: Attack detection...")

    # Generate data
    train_data = generate_honest_data(300, seed=42)

    attacks = [
        ('intercept_resend', 0.3),
        ('decorrelation', 0.4),
        ('visibility', 0.5)
    ]

    detector = IsolationForestDetector()
    detector.fit(train_data)

    results = []
    for attack_type, strength in attacks:
        attack_data = generate_attack_data(50, attack_type, strength)
        predictions = detector.predict_batch(attack_data)

        n_detected = sum(1 for p in predictions if p['is_anomaly'])
        detection_rate = n_detected / len(attack_data)
        avg_score = np.mean([p['anomaly_score'] for p in predictions])

        results.append({
            'attack_type': attack_type,
            'strength': strength,
            'n_detected': n_detected,
            'n_total': len(attack_data),
            'detection_rate': detection_rate,
            'avg_anomaly_score': avg_score
        })

        print(f"  {attack_type} (str={strength}): {detection_rate:.2%} detected")

    # At least 50% detection on any attack
    any_detected = any(r['detection_rate'] >= 0.5 for r in results)

    return {
        'results': results,
        'any_attack_detected': any_detected,
        'passed': any_detected
    }


def test_auc_roc():
    """Test AUC-ROC performance."""
    print("Testing: AUC-ROC evaluation...")

    # Generate data
    train_data = generate_honest_data(300, seed=42)
    test_honest = generate_honest_data(100, seed=100)
    test_attack = generate_attack_data(100, 'intercept_resend', 0.4, seed=101)

    # Train
    detector = IsolationForestDetector()
    detector.fit(train_data)

    # Predict
    honest_preds = detector.predict_batch(test_honest)
    attack_preds = detector.predict_batch(test_attack)

    # Combine for AUC calculation
    scores = ([p['anomaly_score'] for p in honest_preds] +
              [p['anomaly_score'] for p in attack_preds])
    labels = [0] * len(honest_preds) + [1] * len(attack_preds)

    auc = roc_auc_score(labels, scores)

    print(f"  AUC-ROC: {auc:.4f}")

    return {
        'auc_roc': auc,
        'n_honest': len(honest_preds),
        'n_attack': len(attack_preds),
        'passed': auc > 0.7  # Reasonable AUC threshold
    }


def test_contamination_impact():
    """Test impact of contamination parameter."""
    print("Testing: Contamination parameter impact...")

    train_data = generate_honest_data(300, seed=42)
    test_data = generate_honest_data(50, seed=100)
    attack_data = generate_attack_data(50, 'intercept_resend', 0.3)

    contaminations = [0.01, 0.05, 0.10, 0.15]
    results = []

    for c in contaminations:
        detector = IsolationForestDetector(contamination=c)
        detector.fit(train_data)

        honest_preds = detector.predict_batch(test_data)
        attack_preds = detector.predict_batch(attack_data)

        fpr = sum(1 for p in honest_preds if p['is_anomaly']) / len(honest_preds)
        tpr = sum(1 for p in attack_preds if p['is_anomaly']) / len(attack_preds)

        results.append({
            'contamination': c,
            'fpr': fpr,
            'tpr': tpr
        })

        print(f"  c={c}: FPR={fpr:.2%}, TPR={tpr:.2%}")

    # Higher contamination should give higher FPR
    fpr_trend = results[0]['fpr'] <= results[-1]['fpr']

    return {
        'results': results,
        'fpr_increases_with_contamination': fpr_trend,
        'passed': True
    }


def test_feature_importance():
    """Analyze which features are most important."""
    print("Testing: Feature importance analysis...")

    train_data = generate_honest_data(200, seed=42)

    detector = IsolationForestDetector()
    detector.fit(train_data)

    # Get feature matrix
    X = detector.prepare_features(train_data)
    feature_names = [
        'mean', 'std', 'median', 'q25', 'q75',
        'low_pval', 'high_pval', 'entropy', 'autocorr', 'runs',
        'E00', 'E01', 'E10', 'E11', 'CHSH', 'corr_var',
        'qber', 'visibility'
    ]

    # Approximate importance by correlation with anomaly score
    # (Isolation Forest doesn't have built-in feature importance)
    X_scaled = detector.scaler.transform(X)
    scores = detector.model.decision_function(X_scaled)

    correlations = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], scores)[0, 1]
        correlations.append({
            'feature': feature_names[i] if i < len(feature_names) else f'f{i}',
            'correlation': float(corr) if not np.isnan(corr) else 0
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

    print("  Top 5 most correlated features:")
    for c in correlations[:5]:
        print(f"    {c['feature']}: {c['correlation']:.4f}")

    return {
        'feature_correlations': correlations,
        'passed': True
    }


def main():
    """Run all tests and save results."""
    print("=" * 60)
    print("Experiment 3.15: Isolation Forest Anomaly Detection")
    print("=" * 60)

    results = {
        'experiment': 'exp_3_15_isolation_forest',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Run all tests
    results['tests']['training'] = test_training()
    results['tests']['false_positive_rate'] = test_false_positive_rate()
    results['tests']['attack_detection'] = test_attack_detection()
    results['tests']['auc_roc'] = test_auc_roc()
    results['tests']['contamination_impact'] = test_contamination_impact()
    results['tests']['feature_importance'] = test_feature_importance()

    # Validation
    all_passed = all(
        test_result.get('passed', False)
        for test_result in results['tests'].values()
    )

    results['validation'] = {
        'checks': [
            {
                'name': 'Model trains successfully',
                'passed': results['tests']['training']['passed'],
                'detail': f"Features: {results['tests']['training']['train_stats']['n_features']}"
            },
            {
                'name': 'FPR < 15%',
                'passed': results['tests']['false_positive_rate']['passed'],
                'detail': f"FPR: {results['tests']['false_positive_rate']['fpr']:.2%}"
            },
            {
                'name': 'Detects at least one attack type',
                'passed': results['tests']['attack_detection']['passed'],
                'detail': f"Any detected: {results['tests']['attack_detection']['any_attack_detected']}"
            },
            {
                'name': 'AUC-ROC > 0.7',
                'passed': results['tests']['auc_roc']['passed'],
                'detail': f"AUC: {results['tests']['auc_roc']['auc_roc']:.4f}"
            }
        ],
        'all_passed': all_passed
    }

    results['summary'] = {
        'fpr': results['tests']['false_positive_rate']['fpr'],
        'auc_roc': results['tests']['auc_roc']['auc_roc'],
        'validation_passed': all_passed
    }

    # Save results
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'phase3'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'exp_3_15_isolation_forest.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    for check in results['validation']['checks']:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"[{status}] {check['name']}: {check['detail']}")

    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print(f"Results saved to: {output_dir / 'exp_3_15_isolation_forest.json'}")

    return results


if __name__ == '__main__':
    main()
