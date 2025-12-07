#!/usr/bin/env python3
"""
Experiment 3.6: Hybrid Detection Ensemble

Combines multiple anomaly detectors for robust attack detection:
- TARA-MMD (best performer from Exp 3.2)
- Classical tests (AD, CvM, Chi2, KS)
- Sequential tests (CUSUM)

Ensemble strategies:
- Majority voting
- Weighted voting
- Any detection (OR)
- All detection (AND)

Research Questions:
1. Does ensemble improve over best individual detector?
2. Which voting strategy works best?
3. What are the tradeoffs (FPR vs detection rate)?

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.stats import beta, wasserstein_distance
from scipy.spatial.distance import pdist
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import PHASE3_RESULTS
from utils.circuits import IdealBellSimulator
from utils.sifting import sift_keys
from utils.tara_integration import TARAk, qkd_to_tara_format, create_eve_data

PHASE3_RESULTS.mkdir(parents=True, exist_ok=True)


class MMDDetector:
    """Simplified MMD detector for ensemble."""

    def __init__(self, calibration_pvalues: np.ndarray, sigma: float = None):
        self.cal_pvalues = calibration_pvalues
        if sigma is None:
            self.sigma = self._median_heuristic(calibration_pvalues)
        else:
            self.sigma = sigma
        self.K_cal = self._rbf_kernel(calibration_pvalues, calibration_pvalues)

    def _median_heuristic(self, X: np.ndarray) -> float:
        X = X.reshape(-1, 1)
        distances = pdist(X)
        return float(np.median(distances)) + 1e-8 if len(distances) > 0 else 1.0

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y**2, axis=1, keepdims=True)
        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * np.dot(X, Y.T)
        return np.exp(-sq_dist / (2 * self.sigma**2))

    def compute_mmd_squared(self, test_pvalues: np.ndarray) -> float:
        m = len(self.cal_pvalues)
        n = len(test_pvalues)

        K_XX = self.K_cal
        term1 = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))

        K_YY = self._rbf_kernel(test_pvalues, test_pvalues)
        term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))

        K_XY = self._rbf_kernel(self.cal_pvalues, test_pvalues)
        term3 = 2 * np.sum(K_XY) / (m * n)

        return term1 + term2 - term3

    def test(self, test_pvalues: np.ndarray, threshold: float = 0.01) -> dict:
        mmd_sq = self.compute_mmd_squared(test_pvalues)
        return {
            'name': 'MMD',
            'statistic': mmd_sq,
            'detected': mmd_sq > threshold
        }


class WassersteinDetector:
    """Wasserstein distance detector."""

    def __init__(self, calibration_pvalues: np.ndarray):
        self.cal_pvalues = calibration_pvalues
        self.threshold = self._estimate_threshold()

    def _estimate_threshold(self) -> float:
        n = len(self.cal_pvalues)
        if n < 100:
            return 0.1

        distances = []
        for _ in range(50):
            idx = np.random.permutation(n)
            half1 = self.cal_pvalues[idx[:n//2]]
            half2 = self.cal_pvalues[idx[n//2:n//2 + n//2]]
            d = wasserstein_distance(half1, half2)
            distances.append(d)

        return np.percentile(distances, 95) * 1.5

    def test(self, test_pvalues: np.ndarray) -> dict:
        w_dist = wasserstein_distance(test_pvalues, self.cal_pvalues)
        return {
            'name': 'Wasserstein',
            'statistic': w_dist,
            'detected': w_dist > self.threshold
        }


class CUSUMDetector:
    """CUSUM detector for ensemble."""

    def __init__(self, threshold: float = 7.0, allowance: float = 0.05):
        self.threshold = threshold
        self.allowance = allowance

    def test(self, pvalues: np.ndarray) -> dict:
        S_plus = 0.0
        S_minus = 0.0
        max_stat = 0.0

        for x in pvalues:
            S_plus = max(0, S_plus + (x - 0.5) - self.allowance)
            S_minus = max(0, S_minus - (x - 0.5) - self.allowance)
            max_stat = max(max_stat, S_plus, S_minus)

        return {
            'name': 'CUSUM',
            'statistic': max_stat,
            'detected': max_stat > self.threshold
        }


class ClassicalTestDetector:
    """Classical tests for ensemble."""

    def test(self, pvalues: np.ndarray) -> dict:
        # KS test
        ks_stat, ks_pval = stats.kstest(pvalues, 'uniform')

        # Anderson-Darling
        n = len(pvalues)
        sorted_p = np.sort(np.clip(pvalues, 1e-10, 1 - 1e-10))
        i = np.arange(1, n + 1)
        A2 = -n - np.mean((2*i - 1) * (np.log(sorted_p) + np.log(1 - sorted_p[::-1])))

        # Chi-squared
        observed, _ = np.histogram(pvalues, bins=10, range=(0, 1))
        expected = n / 10
        chi2 = np.sum((observed - expected)**2 / expected)
        chi2_pval = 1 - stats.chi2.cdf(chi2, 9)

        ks_det = ks_pval < 0.05
        ad_det = A2 > 2.492
        chi2_det = chi2_pval < 0.05

        return {
            'name': 'Classical',
            'ks_detected': ks_det,
            'ad_detected': ad_det,
            'chi2_detected': chi2_det,
            'detected': any([ks_det, ad_det, chi2_det])  # Any classical test
        }


class HybridEnsemble:
    """Hybrid ensemble detector combining multiple methods."""

    def __init__(self, calibration_pvalues: np.ndarray):
        self.mmd = MMDDetector(calibration_pvalues)
        self.wasserstein = WassersteinDetector(calibration_pvalues)
        self.cusum = CUSUMDetector()
        self.classical = ClassicalTestDetector()

    def test(self, test_pvalues: np.ndarray, voting: str = 'majority') -> dict:
        results = {
            'mmd': self.mmd.test(test_pvalues),
            'wasserstein': self.wasserstein.test(test_pvalues),
            'cusum': self.cusum.test(test_pvalues),
            'classical': self.classical.test(test_pvalues)
        }

        detections = [r['detected'] for r in results.values()]

        if voting == 'majority':
            combined = sum(detections) > len(detections) / 2
        elif voting == 'any':
            combined = any(detections)
        elif voting == 'all':
            combined = all(detections)
        else:  # weighted - give more weight to MMD
            weights = [2.0, 1.0, 1.0, 1.0]  # MMD gets double weight
            weighted_sum = sum(w * d for w, d in zip(weights, detections))
            combined = weighted_sum >= sum(weights) / 2

        return {
            'individual_results': results,
            'n_detected': sum(detections),
            'n_detectors': len(detections),
            'combined_detected': combined,
            'voting': voting
        }


def compute_pvalues(test_data: dict, tara_k: TARAk) -> np.ndarray:
    """Compute conformal p-values for test data."""
    pvalues = []
    for i in range(len(test_data['x'])):
        p = tara_k.cond_probs[(test_data['x'][i], test_data['y'][i])][test_data['a'][i], test_data['b'][i]]
        score = -np.log(p + 1e-10)
        rank = np.sum(tara_k.cal_scores >= score) + 1
        pvalues.append(rank / (len(tara_k.cal_scores) + 1))
    return np.array(pvalues)


def run_fpr_test(seed: int = 42) -> dict:
    """Test 1: False positive rate for different voting strategies.

    CORRECTED: Uses REAL TARA p-values from QKD simulation, not synthetic.
    NOTE: Classical tests expected to have high FPR due to p-value non-uniformity.
    """
    print("\n" + "=" * 60)
    print("TEST 1: FALSE POSITIVE RATE BY VOTING STRATEGY")
    print("(Using REAL TARA p-values from QKD simulation)")
    print("=" * 60)

    n_trials = 25
    voting_strategies = ['majority', 'any', 'all', 'weighted']
    fpr = {v: 0 for v in voting_strategies}
    individual_fpr = {'mmd': 0, 'wasserstein': 0, 'cusum': 0, 'classical': 0}

    for i in range(n_trials):
        trial_seed = seed + i * 100

        # CORRECTED: Generate REAL TARA p-values from honest QKD simulation
        # (not synthetic np.random.uniform which hides non-uniformity issues)
        cal_sim = IdealBellSimulator(visibility=1.0, seed=trial_seed)
        cal_qkd = cal_sim.generate_qkd_data(n_pairs=3000, key_fraction=0.5)
        cal_sift = sift_keys(cal_qkd)
        cal_data = qkd_to_tara_format(cal_sift.security_data)

        tara_k = TARAk(cal_data)
        cal_pvalues = tara_k.compute_p_values(cal_data)  # Real TARA p-values

        # Generate independent test data (also honest)
        test_sim = IdealBellSimulator(visibility=1.0, seed=trial_seed + 5000)
        test_qkd = test_sim.generate_qkd_data(n_pairs=2000, key_fraction=0.5)
        test_sift = sift_keys(test_qkd)
        test_data = qkd_to_tara_format(test_sift.security_data)

        test_pvalues = tara_k.compute_p_values(test_data)  # Real TARA p-values

        ensemble = HybridEnsemble(cal_pvalues)

        for voting in voting_strategies:
            result = ensemble.test(test_pvalues, voting=voting)
            if result['combined_detected']:
                fpr[voting] += 1

            if voting == 'majority':  # Count individual FPs once
                for det_name, det_result in result['individual_results'].items():
                    if det_result['detected']:
                        individual_fpr[det_name] += 1

    print("\nFPR by Voting Strategy:")
    for voting, count in fpr.items():
        rate = 100 * count / n_trials
        print(f"  {voting}: {rate:.1f}%")

    print("\nIndividual Detector FPR:")
    for det, count in individual_fpr.items():
        rate = 100 * count / n_trials
        print(f"  {det}: {rate:.1f}%")

    return {
        'n_trials': n_trials,
        'voting_fpr': {v: c/n_trials for v, c in fpr.items()},
        'individual_fpr': {d: c/n_trials for d, c in individual_fpr.items()}
    }


def run_detection_test(seed: int = 42) -> dict:
    """Test 2: Attack detection comparison."""
    print("\n" + "=" * 60)
    print("TEST 2: ATTACK DETECTION COMPARISON")
    print("=" * 60)

    # Generate calibration data
    sim = IdealBellSimulator(visibility=1.0, seed=seed)
    cal_qkd = sim.generate_qkd_data(n_pairs=5000, key_fraction=0.5)
    cal_sift = sift_keys(cal_qkd)
    cal_data = qkd_to_tara_format(cal_sift.security_data)

    tara_k = TARAk(cal_data)
    cal_pvalues = compute_pvalues(cal_data, tara_k)

    ensemble = HybridEnsemble(cal_pvalues)

    scenarios = [
        ('Honest', 'none', 0.0, 1.0),
        ('Intercept 0.2', 'intercept_resend', 0.2, 1.0),
        ('Intercept 0.4', 'intercept_resend', 0.4, 1.0),
        ('Decorrelation 0.3', 'decorrelation', 0.3, 1.0),
        ('Visibility 0.85', 'visibility', 0.0, 0.85),
        ('Visibility 0.75', 'visibility', 0.0, 0.75),
    ]

    results = []
    voting_strategies = ['majority', 'any', 'weighted']

    print(f"\n{'Scenario':<20} {'MMD':<8} {'W':<8} {'CUSUM':<8} {'Class':<8} | {'Maj':<8} {'Any':<8} {'Wtd':<8}")
    print("-" * 88)

    for scenario_name, attack_type, noise_level, visibility in scenarios:
        test_sim = IdealBellSimulator(visibility=visibility, seed=seed + 1000)
        test_qkd = test_sim.generate_qkd_data(n_pairs=2000, key_fraction=0.5)
        test_sift = sift_keys(test_qkd)
        test_data = qkd_to_tara_format(test_sift.security_data)

        if attack_type == 'intercept_resend':
            test_data = create_eve_data(test_data, attack_type='intercept_resend',
                                         noise_level=noise_level, seed=seed + 2000)
        elif attack_type == 'decorrelation':
            test_data = create_eve_data(test_data, attack_type='decorrelation',
                                         noise_level=noise_level, seed=seed + 2000)

        test_pvalues = compute_pvalues(test_data, tara_k)

        row = {'scenario': scenario_name, 'attack_type': attack_type}

        ind_results = {}
        for voting in voting_strategies:
            result = ensemble.test(test_pvalues, voting=voting)
            row[f'{voting}_detected'] = result['combined_detected']

            if voting == 'majority':
                ind_results = result['individual_results']

        mmd = "DET" if ind_results['mmd']['detected'] else "-"
        w = "DET" if ind_results['wasserstein']['detected'] else "-"
        cusum = "DET" if ind_results['cusum']['detected'] else "-"
        classical = "DET" if ind_results['classical']['detected'] else "-"
        maj = "DET" if row['majority_detected'] else "-"
        any_d = "DET" if row['any_detected'] else "-"
        wtd = "DET" if row['weighted_detected'] else "-"

        print(f"{scenario_name:<20} {mmd:<8} {w:<8} {cusum:<8} {classical:<8} | {maj:<8} {any_d:<8} {wtd:<8}")

        row['individual'] = {
            'mmd': ind_results['mmd']['detected'],
            'wasserstein': ind_results['wasserstein']['detected'],
            'cusum': ind_results['cusum']['detected'],
            'classical': ind_results['classical']['detected']
        }
        results.append(row)

    # Summary
    n_attacks = sum(1 for r in results if r['attack_type'] != 'none')
    print("\n--- Detection Summary ---")
    for voting in voting_strategies:
        count = sum(1 for r in results if r[f'{voting}_detected'] and r['attack_type'] != 'none')
        print(f"{voting}: {count}/{n_attacks}")

    # Individual
    for det in ['mmd', 'wasserstein', 'cusum', 'classical']:
        count = sum(1 for r in results if r['individual'][det] and r['attack_type'] != 'none')
        print(f"{det}: {count}/{n_attacks}")

    return {
        'results': results,
        'n_attacks': n_attacks
    }


def run_robustness_test(seed: int = 42) -> dict:
    """Test 3: Robustness - does ensemble beat best individual?

    CORRECTED: Uses real TARA p-values for calibration to reflect actual behavior.
    """
    print("\n" + "=" * 60)
    print("TEST 3: ENSEMBLE VS BEST INDIVIDUAL")
    print("(Using REAL TARA calibration p-values)")
    print("=" * 60)

    # CORRECTED: Generate real TARA calibration p-values
    cal_sim = IdealBellSimulator(visibility=1.0, seed=seed)
    cal_qkd = cal_sim.generate_qkd_data(n_pairs=3000, key_fraction=0.5)
    cal_sift = sift_keys(cal_qkd)
    cal_data = qkd_to_tara_format(cal_sift.security_data)
    tara_k = TARAk(cal_data)
    cal_pvalues = tara_k.compute_p_values(cal_data)
    ensemble = HybridEnsemble(cal_pvalues)

    # Different attack types
    attack_configs = [
        ('Shift', lambda: beta.rvs(2.0, 1, size=300)),
        ('Heavy Shift', lambda: beta.rvs(3.0, 1, size=300)),
        ('Bimodal', lambda: np.concatenate([
            np.random.uniform(0, 0.3, 150),
            np.random.uniform(0.7, 1.0, 150)
        ])),
    ]

    results = []
    for attack_name, attack_fn in attack_configs:
        np.random.seed(seed)
        test_pvalues = attack_fn()

        result = ensemble.test(test_pvalues, voting='weighted')

        detections = result['individual_results']
        best_individual = max([
            ('MMD', detections['mmd']['detected']),
            ('W', detections['wasserstein']['detected']),
            ('CUSUM', detections['cusum']['detected']),
            ('Classical', detections['classical']['detected'])
        ], key=lambda x: x[1])

        ensemble_det = result['combined_detected']

        print(f"\n{attack_name}:")
        print(f"  MMD: {'DET' if detections['mmd']['detected'] else '-'}")
        print(f"  Wasserstein: {'DET' if detections['wasserstein']['detected'] else '-'}")
        print(f"  CUSUM: {'DET' if detections['cusum']['detected'] else '-'}")
        print(f"  Classical: {'DET' if detections['classical']['detected'] else '-'}")
        print(f"  Ensemble: {'DET' if ensemble_det else '-'}")

        results.append({
            'attack': attack_name,
            'mmd': detections['mmd']['detected'],
            'wasserstein': detections['wasserstein']['detected'],
            'cusum': detections['cusum']['detected'],
            'classical': detections['classical']['detected'],
            'ensemble': ensemble_det
        })

    return {'results': results}


def validate_results(all_results: dict) -> dict:
    """Validate experiment results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: Majority voting has acceptable FPR
    fpr = all_results.get('fpr', {})
    voting_fpr = fpr.get('voting_fpr', {})
    maj_fpr = voting_fpr.get('majority', 1.0)
    checks.append({
        'name': 'Majority voting FPR < 20%',
        'passed': maj_fpr < 0.20,
        'detail': f"FPR: {100*maj_fpr:.1f}%"
    })

    # Check 2: Ensemble detects attacks
    detection = all_results.get('detection', {})
    results = detection.get('results', [])
    attacks = [r for r in results if r['attack_type'] != 'none']
    any_detected = any(r['any_detected'] for r in attacks)
    checks.append({
        'name': 'Ensemble detects at least 1 attack',
        'passed': any_detected,
        'detail': f"Attack scenarios: {len(attacks)}"
    })

    # Check 3: 'Any' strategy has high detection
    any_count = sum(1 for r in attacks if r['any_detected'])
    checks.append({
        'name': "'Any' strategy detection >= 50%",
        'passed': any_count >= len(attacks) / 2 if attacks else True,
        'detail': f"Detection: {any_count}/{len(attacks)}"
    })

    # Print validation
    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    n_passed = sum(1 for c in checks if c['passed'])
    print(f"\nPassed: {n_passed}/{len(checks)} checks")
    print(f"Overall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete Hybrid Ensemble experiment."""
    print("=" * 60)
    print("EXPERIMENT 3.6: HYBRID DETECTION ENSEMBLE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_3_6_hybrid_ensemble',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1
    print("\n" + "#" * 60)
    print("RUNNING TEST 1: FPR")
    print("#" * 60)
    fpr_results = run_fpr_test(seed=42)
    all_results['tests']['fpr'] = fpr_results

    # Test 2
    print("\n" + "#" * 60)
    print("RUNNING TEST 2: DETECTION")
    print("#" * 60)
    detection_results = run_detection_test(seed=42)
    all_results['tests']['detection'] = detection_results

    # Test 3
    print("\n" + "#" * 60)
    print("RUNNING TEST 3: ROBUSTNESS")
    print("#" * 60)
    robustness_results = run_robustness_test(seed=42)
    all_results['tests']['robustness'] = robustness_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"\nMajority FPR: {100*fpr_results['voting_fpr']['majority']:.1f}%")
    print(f"Any FPR: {100*fpr_results['voting_fpr']['any']:.1f}%")
    print(f"Validation passed: {validation['all_passed']}")

    all_results['summary'] = {
        'majority_fpr': fpr_results['voting_fpr']['majority'],
        'any_fpr': fpr_results['voting_fpr']['any'],
        'validation_passed': validation['all_passed']
    }

    # Save
    output_file = PHASE3_RESULTS / 'exp_3_6_hybrid_ensemble.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    return all_results


if __name__ == "__main__":
    results = run_experiment()
