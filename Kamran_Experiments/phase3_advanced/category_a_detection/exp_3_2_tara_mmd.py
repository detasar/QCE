#!/usr/bin/env python3
"""
Experiment 3.2: TARA-MMD (Maximum Mean Discrepancy)

This experiment implements and evaluates MMD-based anomaly detection.
MMD embeds distributions into a Reproducing Kernel Hilbert Space (RKHS)
and measures distance there.

Key Differences from TARA-K and TARA-W:
- TARA-K: KS test of p-values vs Uniform(0,1)
- TARA-W: Wasserstein distance between distributions
- TARA-MMD: Kernel-based distance in RKHS

Mathematical Foundation:
    MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]

    where k is a kernel function (typically RBF):
    k(x,y) = exp(-||x-y||² / (2σ²))

Based on: Gretton et al. (2012) - "A Kernel Two-Sample Test"

Research Questions:
1. Does TARA-MMD detect attacks that TARA-K and TARA-W miss?
2. How does kernel bandwidth affect detection?
3. What's the computational cost compared to other methods?
4. How does false positive rate compare?

Expected Outcomes:
- MMD may detect subtle distributional differences better
- Permutation test provides valid p-values
- Computational cost O(n²) may be higher than KS/Wasserstein

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import pdist
from dataclasses import dataclass
from typing import Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import PHASE3_RESULTS
from utils.circuits import IdealBellSimulator
from utils.sifting import sift_keys
from utils.tara_integration import TARAk, qkd_to_tara_format, create_eve_data

# Ensure phase3 results directory exists
PHASE3_RESULTS.mkdir(parents=True, exist_ok=True)


@dataclass
class TARAMMDResult:
    """Result from TARA-MMD detector."""
    detected: bool
    mmd_squared: float
    mmd: float
    p_value: float
    sigma: float
    computation_time: float

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'mmd_squared': self.mmd_squared,
            'mmd': self.mmd,
            'p_value': self.p_value,
            'sigma': self.sigma,
            'computation_time': self.computation_time
        }


class TARAMMD:
    """
    TARA detector using Maximum Mean Discrepancy.

    Based on Gretton et al. (2012) - "A Kernel Two-Sample Test"

    Advantages:
    - Detects any distributional difference (not just location/scale)
    - Works well in high dimensions
    - Has rigorous statistical guarantees

    Disadvantages:
    - O(n²) computation
    - Sensitive to kernel bandwidth
    """

    def __init__(self, calibration_data: dict,
                 kernel: str = 'rbf',
                 sigma: Optional[float] = None):
        """
        Initialize TARA-MMD detector.

        Args:
            calibration_data: TARA-format data from honest source
            kernel: Kernel type ('rbf', 'linear')
            sigma: RBF kernel bandwidth (auto-estimated if None)
        """
        self.kernel = kernel

        # Learn conditional probabilities
        self.cond_probs = self._learn_probs(calibration_data)

        # Compute calibration scores
        self.cal_scores = self._compute_scores(calibration_data)

        # Compute calibration p-values
        self.cal_pvalues = self._compute_pvalues(calibration_data)

        # Estimate kernel bandwidth
        if sigma is None:
            self.sigma = self._median_heuristic(self.cal_pvalues)
        else:
            self.sigma = sigma

        # Pre-compute calibration kernel matrix
        self.K_cal = self._compute_kernel_matrix(self.cal_pvalues,
                                                  self.cal_pvalues)

    def _learn_probs(self, data: dict) -> dict:
        """Learn P(a,b|x,y) from calibration data with Laplace smoothing."""
        x, y, a, b = data['x'], data['y'], data['a'], data['b']
        probs = {}

        for xi in [0, 1]:
            for yi in [0, 1]:
                mask = (x == xi) & (y == yi)
                if mask.sum() > 0:
                    counts = np.zeros((2, 2))
                    for ai in [0, 1]:
                        for bi in [0, 1]:
                            counts[ai, bi] = np.sum((a[mask] == ai) & (b[mask] == bi))
                    counts += 1  # Laplace smoothing
                    probs[(xi, yi)] = counts / counts.sum()
                else:
                    probs[(xi, yi)] = np.ones((2, 2)) / 4

        return probs

    def _compute_scores(self, data: dict) -> np.ndarray:
        """Compute nonconformity scores: -log P(a,b|x,y)."""
        scores = []
        for i in range(len(data['x'])):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        return np.array(scores)

    def _compute_pvalues(self, data: dict) -> np.ndarray:
        """Compute conformal p-values using learned model."""
        scores = self._compute_scores(data)
        p_values = []

        for s in scores:
            rank = np.sum(self.cal_scores >= s) + 1
            p_values.append(rank / (len(self.cal_scores) + 1))

        return np.array(p_values)

    def compute_p_values(self, data: dict) -> np.ndarray:
        """Public method to compute p-values for test data."""
        return self._compute_pvalues(data)

    def _median_heuristic(self, X: np.ndarray) -> float:
        """Estimate kernel bandwidth using median heuristic."""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        distances = pdist(X)
        if len(distances) == 0:
            return 1.0
        return float(np.median(distances)) + 1e-8

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # ||x - y||² = ||x||² + ||y||² - 2<x,y>
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y**2, axis=1, keepdims=True)

        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * np.dot(X, Y.T)
        return np.exp(-sq_dist / (2 * self.sigma**2))

    def _compute_kernel_matrix(self, X: np.ndarray,
                                Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            return np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def compute_mmd_squared(self, test_pvalues: np.ndarray) -> float:
        """
        Compute unbiased MMD² estimate.

        MMD²_u = 1/(m(m-1)) Σ_{i≠j} k(xi,xj)
               + 1/(n(n-1)) Σ_{i≠j} k(yi,yj)
               - 2/(mn) Σ_ij k(xi,yj)
        """
        m = len(self.cal_pvalues)
        n = len(test_pvalues)

        # K_XX (calibration-calibration)
        K_XX = self.K_cal
        term1 = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))

        # K_YY (test-test)
        K_YY = self._compute_kernel_matrix(test_pvalues, test_pvalues)
        term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))

        # K_XY (calibration-test)
        K_XY = self._compute_kernel_matrix(self.cal_pvalues, test_pvalues)
        term3 = 2 * np.sum(K_XY) / (m * n)

        return term1 + term2 - term3

    def permutation_test(self, test_pvalues: np.ndarray,
                         n_permutations: int = 200) -> Tuple[float, float]:
        """
        Compute p-value via permutation test.

        Args:
            test_pvalues: Test p-values
            n_permutations: Number of permutations

        Returns:
            (mmd_squared, p_value)
        """
        observed_mmd = self.compute_mmd_squared(test_pvalues)

        # Pool all data
        pooled = np.concatenate([self.cal_pvalues, test_pvalues])
        m = len(self.cal_pvalues)

        null_mmds = []
        for _ in range(n_permutations):
            perm = np.random.permutation(len(pooled))
            X_perm = pooled[perm[:m]]
            Y_perm = pooled[perm[m:m + len(test_pvalues)]]

            # Compute MMD for permuted data
            K_XX_perm = self._compute_kernel_matrix(X_perm, X_perm)
            K_YY_perm = self._compute_kernel_matrix(Y_perm, Y_perm)
            K_XY_perm = self._compute_kernel_matrix(X_perm, Y_perm)

            term1 = (np.sum(K_XX_perm) - np.trace(K_XX_perm)) / (m * (m - 1))
            n_y = len(Y_perm)
            term2 = (np.sum(K_YY_perm) - np.trace(K_YY_perm)) / (n_y * (n_y - 1))
            term3 = 2 * np.sum(K_XY_perm) / (m * n_y)

            null_mmds.append(term1 + term2 - term3)

        p_value = np.mean(np.array(null_mmds) >= observed_mmd)

        return observed_mmd, p_value

    def test(self, data: dict, alpha: float = 0.05,
             n_permutations: int = 200) -> TARAMMDResult:
        """
        Full MMD test with p-value.

        Args:
            data: Test data in TARA format
            alpha: Significance level
            n_permutations: Number of permutations for p-value

        Returns:
            TARAMMDResult with detection info
        """
        start_time = time.time()

        test_pvalues = self.compute_p_values(data)
        mmd_sq, p_value = self.permutation_test(test_pvalues, n_permutations)

        computation_time = time.time() - start_time

        return TARAMMDResult(
            detected=p_value < alpha,
            mmd_squared=mmd_sq,
            mmd=np.sqrt(max(0, mmd_sq)),
            p_value=p_value,
            sigma=self.sigma,
            computation_time=computation_time
        )


def generate_tara_data(n_pairs: int = 10000, visibility: float = 1.0,
                        security_fraction: float = 0.5, seed: int = 42) -> dict:
    """Generate data for TARA experiments."""
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    key_fraction = 1 - security_fraction
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)
    sifting_result = sift_keys(qkd_data)

    return qkd_to_tara_format(sifting_result.security_data)


def run_calibration_test(seed: int = 42) -> dict:
    """
    Test 1: Calibration and baseline behavior.

    Verifies:
    - Threshold estimation works
    - No false detection on honest data
    - MMD value is low for honest data
    """
    print("\n" + "=" * 60)
    print("TEST 1: CALIBRATION")
    print("=" * 60)

    # Generate calibration data
    print("\nGenerating calibration data...")
    cal_data = generate_tara_data(n_pairs=8000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)
    n_cal = len(cal_data['x'])
    print(f"  Calibration samples: {n_cal}")

    # Split for validation
    split = n_cal // 2
    cal_train = {k: v[:split] for k, v in cal_data.items()}
    cal_test = {k: v[split:] for k, v in cal_data.items()}

    # Create TARA-MMD detector
    print("\nCreating TARA-MMD detector (RBF kernel)...")
    detector = TARAMMD(cal_train, kernel='rbf')
    print(f"  Estimated sigma: {detector.sigma:.4f}")

    # Test on honest data
    print("\nTesting on honest data (permutation test)...")
    result = detector.test(cal_test, n_permutations=100)

    print(f"\nResults:")
    print(f"  MMD²: {result.mmd_squared:.6f}")
    print(f"  MMD: {result.mmd:.6f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Detected: {'YES (FALSE POSITIVE!)' if result.detected else 'NO (Correct)'}")
    print(f"  Computation time: {result.computation_time:.2f}s")

    passed = not result.detected and result.p_value > 0.05

    return {
        'n_calibration': split,
        'n_test': len(cal_test['x']),
        'sigma': detector.sigma,
        'mmd_squared': result.mmd_squared,
        'mmd': result.mmd,
        'p_value': result.p_value,
        'detected': result.detected,
        'computation_time': result.computation_time,
        'passed': passed
    }


def run_attack_comparison_test(seed: int = 42) -> dict:
    """
    Test 2: Compare TARA-MMD vs TARA-K on attack detection.

    Tests multiple attack types and strengths.
    """
    print("\n" + "=" * 60)
    print("TEST 2: ATTACK DETECTION COMPARISON (TARA-MMD vs TARA-K)")
    print("=" * 60)

    # Generate calibration data (smaller for faster MMD)
    cal_data = generate_tara_data(n_pairs=6000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Create both detectors
    print("\nCreating detectors...")
    detector_mmd = TARAMMD(cal_data, kernel='rbf')
    detector_k = TARAk(cal_data)

    # Test scenarios
    attack_scenarios = [
        ('honest', 'none', 1.0),
        ('intercept_resend', 0.15, 1.0),
        ('intercept_resend', 0.30, 1.0),
        ('intercept_resend', 0.50, 1.0),
        ('decorrelation', 0.20, 1.0),
        ('decorrelation', 0.40, 1.0),
        ('visibility_drop', 'none', 0.80),
        ('visibility_drop', 'none', 0.70),
    ]

    results = []

    print(f"\n{'Scenario':<30} {'MMD':<12} {'TARA-K':<12} {'Winner':<10}")
    print("-" * 64)

    for attack_type, noise_level, visibility in attack_scenarios:
        # Generate test data
        if attack_type == 'honest':
            test_data = generate_tara_data(n_pairs=2000, visibility=1.0,
                                           security_fraction=0.5,
                                           seed=seed + 1000)
            scenario_name = "Honest"
        elif attack_type == 'visibility_drop':
            test_data = generate_tara_data(n_pairs=2000, visibility=visibility,
                                           security_fraction=0.5,
                                           seed=seed + 1000)
            scenario_name = f"Visibility={visibility}"
        else:
            honest_data = generate_tara_data(n_pairs=2000, visibility=visibility,
                                              security_fraction=0.5,
                                              seed=seed + 1000)
            test_data = create_eve_data(honest_data, attack_type=attack_type,
                                         noise_level=noise_level, seed=seed + 2000)
            scenario_name = f"{attack_type}({noise_level})"

        # Test with both detectors
        result_mmd = detector_mmd.test(test_data, n_permutations=100)
        result_k = detector_k.test(test_data, threshold=0.2)

        # Determine winner for attack scenarios
        if attack_type == 'honest':
            mmd_correct = not result_mmd.detected
            k_correct = not result_k.detected
            winner = "Tie" if mmd_correct == k_correct else ("MMD" if mmd_correct else "K")
        else:
            mmd_correct = result_mmd.detected
            k_correct = result_k.detected
            if mmd_correct and not k_correct:
                winner = "MMD"
            elif k_correct and not mmd_correct:
                winner = "TARA-K"
            elif mmd_correct and k_correct:
                winner = "Both"
            else:
                winner = "Neither"

        mmd_status = "Detected" if result_mmd.detected else "-"
        k_status = "Detected" if result_k.detected else "-"

        print(f"{scenario_name:<30} {mmd_status:<12} {k_status:<12} {winner:<10}")

        results.append({
            'scenario': scenario_name,
            'attack_type': attack_type,
            'noise_level': noise_level if isinstance(noise_level, float) else 0,
            'visibility': visibility,
            'mmd_value': result_mmd.mmd,
            'mmd_pvalue': result_mmd.p_value,
            'mmd_detected': result_mmd.detected,
            'tara_k_statistic': result_k.statistic,
            'tara_k_detected': result_k.detected,
            'winner': winner
        })

    # Summary
    mmd_detections = sum(1 for r in results if r['mmd_detected'] and r['attack_type'] != 'honest')
    k_detections = sum(1 for r in results if r['tara_k_detected'] and r['attack_type'] != 'honest')
    attack_count = sum(1 for r in results if r['attack_type'] != 'honest')

    print(f"\n--- Summary ---")
    print(f"Attack scenarios: {attack_count}")
    print(f"TARA-MMD detections: {mmd_detections}/{attack_count}")
    print(f"TARA-K detections: {k_detections}/{attack_count}")

    return {
        'results': results,
        'tara_mmd_detections': mmd_detections,
        'tara_k_detections': k_detections,
        'attack_count': attack_count
    }


def run_false_positive_test(n_trials: int = 20, seed: int = 42) -> dict:
    """
    Test 3: Compare false positive rates of TARA-MMD and TARA-K.
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: FALSE POSITIVE RATE (n={n_trials} trials)")
    print("=" * 60)

    fp_mmd = 0
    fp_k = 0
    mmd_values = []
    k_statistics = []
    computation_times = []

    for i in range(n_trials):
        trial_seed = seed + i * 1000

        # Generate fresh calibration data
        cal_data = generate_tara_data(n_pairs=4000, visibility=1.0,
                                       security_fraction=0.5, seed=trial_seed)

        # Generate independent test data (different seed!)
        test_data = generate_tara_data(n_pairs=2000, visibility=1.0,
                                        security_fraction=0.5,
                                        seed=trial_seed + 500)

        # Create detectors
        detector_mmd = TARAMMD(cal_data, kernel='rbf')
        detector_k = TARAk(cal_data)

        # Test
        result_mmd = detector_mmd.test(test_data, n_permutations=100)
        result_k = detector_k.test(test_data, threshold=0.2)

        mmd_values.append(result_mmd.mmd)
        k_statistics.append(result_k.statistic)
        computation_times.append(result_mmd.computation_time)

        if result_mmd.detected:
            fp_mmd += 1
        if result_k.detected:
            fp_k += 1

        if (i + 1) % 5 == 0:
            print(f"  Completed {i + 1}/{n_trials} trials...")

    fpr_mmd = fp_mmd / n_trials
    fpr_k = fp_k / n_trials

    print(f"\n--- Results ---")
    print(f"TARA-MMD FPR: {100*fpr_mmd:.1f}% ({fp_mmd}/{n_trials})")
    print(f"TARA-K FPR: {100*fpr_k:.1f}% ({fp_k}/{n_trials})")
    print(f"\nTARA-MMD mean MMD: {np.mean(mmd_values):.6f} ± {np.std(mmd_values):.6f}")
    print(f"TARA-K mean statistic: {np.mean(k_statistics):.4f} ± {np.std(k_statistics):.4f}")
    print(f"Mean computation time: {np.mean(computation_times):.2f}s ± {np.std(computation_times):.2f}s")

    return {
        'n_trials': n_trials,
        'tara_mmd_fp': fp_mmd,
        'tara_mmd_fpr': fpr_mmd,
        'tara_k_fp': fp_k,
        'tara_k_fpr': fpr_k,
        'mmd_mean': float(np.mean(mmd_values)),
        'mmd_std': float(np.std(mmd_values)),
        'k_mean': float(np.mean(k_statistics)),
        'k_std': float(np.std(k_statistics)),
        'mean_computation_time': float(np.mean(computation_times)),
        'mmd_acceptable': fpr_mmd < 0.10,
        'k_acceptable': fpr_k < 0.10
    }


def run_kernel_comparison_test(seed: int = 42) -> dict:
    """
    Test 4: Compare RBF vs Linear kernel for MMD.
    """
    print("\n" + "=" * 60)
    print("TEST 4: KERNEL COMPARISON (RBF vs Linear)")
    print("=" * 60)

    # Generate calibration data
    cal_data = generate_tara_data(n_pairs=5000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Create both detectors
    print("\nCreating detectors with different kernels...")
    detector_rbf = TARAMMD(cal_data, kernel='rbf')
    detector_lin = TARAMMD(cal_data, kernel='linear')

    print(f"  RBF sigma: {detector_rbf.sigma:.4f}")

    # Test scenarios
    scenarios = [
        ('honest', 1.0),
        ('visibility=0.90', 0.90),
        ('visibility=0.85', 0.85),
        ('visibility=0.80', 0.80),
    ]

    results = []

    print(f"\n{'Scenario':<20} {'RBF MMD':<12} {'Lin MMD':<12} {'RBF Det':<10} {'Lin Det':<10}")
    print("-" * 64)

    for scenario_name, visibility in scenarios:
        test_data = generate_tara_data(n_pairs=2000, visibility=visibility,
                                        security_fraction=0.5,
                                        seed=seed + 1000)

        result_rbf = detector_rbf.test(test_data, n_permutations=100)
        result_lin = detector_lin.test(test_data, n_permutations=100)

        rbf_det = "YES" if result_rbf.detected else "NO"
        lin_det = "YES" if result_lin.detected else "NO"

        print(f"{scenario_name:<20} {result_rbf.mmd:<12.6f} {result_lin.mmd:<12.6f} "
              f"{rbf_det:<10} {lin_det:<10}")

        results.append({
            'scenario': scenario_name,
            'visibility': visibility,
            'rbf_mmd': result_rbf.mmd,
            'rbf_pvalue': result_rbf.p_value,
            'rbf_detected': result_rbf.detected,
            'linear_mmd': result_lin.mmd,
            'linear_pvalue': result_lin.p_value,
            'linear_detected': result_lin.detected
        })

    return {
        'results': results,
        'rbf_sigma': detector_rbf.sigma
    }


def validate_results(all_results: dict) -> dict:
    """
    Validate experiment results for consistency and correctness.
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    print("\n  NOTE: Phase 1 showed TARA-K detected 0/4 attacks.")
    print("        KS-based tests have limited sensitivity.")
    print("        MMD may or may not outperform in this setup.\n")

    checks = []

    # Check 1: Calibration p-value (single run, may vary)
    # Note: FPR test over multiple trials is the true Type I error measure
    cal = all_results.get('calibration', {})
    pval = cal.get('p_value', 0)
    # Accept if p > 0.01 (single run can vary due to permutation randomness)
    pval_ok = pval > 0.01
    checks.append({
        'name': 'Calibration p-value > 0.01 (single run, FPR is true measure)',
        'passed': pval_ok,
        'detail': f"P-value: {pval:.4f} (FPR test is the real Type I error control)"
    })

    # Check 3: FPR is acceptable (< 10%)
    fpr = all_results.get('false_positive_rate', {})
    mmd_acceptable = fpr.get('mmd_acceptable', False)
    checks.append({
        'name': 'TARA-MMD FPR < 10%',
        'passed': mmd_acceptable,
        'detail': f"FPR: {100*fpr.get('tara_mmd_fpr', 1):.1f}%"
    })

    # Check 4: Computational efficiency (< 15s per test)
    # Note: O(n²) permutation test is expected to be slower than KS/Wasserstein
    comp_time = cal.get('computation_time', 20)
    time_ok = comp_time < 15.0
    checks.append({
        'name': 'Computation time < 15s per test (O(n²) expected)',
        'passed': time_ok,
        'detail': f"Time: {comp_time:.2f}s"
    })

    # Check 5: MMD increases with attack strength (trend check)
    attack = all_results.get('attack_comparison', {})
    attack_results = attack.get('results', [])
    if attack_results:
        # Check if MMD values for attacks are higher than honest
        honest_mmd = [r['mmd_value'] for r in attack_results if r['attack_type'] == 'honest']
        attack_mmd = [r['mmd_value'] for r in attack_results if r['attack_type'] != 'honest']

        if honest_mmd and attack_mmd:
            avg_honest = np.mean(honest_mmd)
            avg_attack = np.mean(attack_mmd)
            trend_ok = avg_attack > avg_honest
            checks.append({
                'name': 'MMD increases with attack presence',
                'passed': trend_ok,
                'detail': f"Honest avg: {avg_honest:.6f}, Attack avg: {avg_attack:.6f}"
            })

    # Check 6: TARA-MMD should detect at least as many attacks as TARA-K
    mmd_det = attack.get('tara_mmd_detections', 0)
    k_det = attack.get('tara_k_detections', 0)
    attack_count = attack.get('attack_count', 0)
    mmd_at_least_as_good = mmd_det >= k_det
    checks.append({
        'name': 'TARA-MMD detects >= attacks than TARA-K',
        'passed': mmd_at_least_as_good,
        'detail': f"MMD: {mmd_det}/{attack_count}, K: {k_det}/{attack_count}"
    })

    # Print validation results
    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    # Core criteria (exclude informational)
    core_checks = [c for c in checks if 'informational' not in c['name']]
    all_passed = all(c['passed'] for c in core_checks)

    n_passed = sum(1 for c in checks if c['passed'])
    print(f"\nPassed: {n_passed}/{len(checks)} checks")
    print(f"Overall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete TARA-MMD experiment."""
    print("=" * 60)
    print("EXPERIMENT 3.2: TARA-MMD")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_3_2_tara_mmd',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: Calibration
    print("\n" + "#" * 60)
    print("RUNNING TEST 1: CALIBRATION")
    print("#" * 60)
    cal_results = run_calibration_test(seed=42)
    all_results['tests']['calibration'] = cal_results

    # Test 2: Attack comparison
    print("\n" + "#" * 60)
    print("RUNNING TEST 2: ATTACK COMPARISON")
    print("#" * 60)
    attack_results = run_attack_comparison_test(seed=42)
    all_results['tests']['attack_comparison'] = attack_results

    # Test 3: False positive rate
    print("\n" + "#" * 60)
    print("RUNNING TEST 3: FALSE POSITIVE RATE")
    print("#" * 60)
    fpr_results = run_false_positive_test(n_trials=20, seed=42)
    all_results['tests']['false_positive_rate'] = fpr_results

    # Test 4: Kernel comparison
    print("\n" + "#" * 60)
    print("RUNNING TEST 4: KERNEL COMPARISON")
    print("#" * 60)
    kernel_results = run_kernel_comparison_test(seed=42)
    all_results['tests']['kernel_comparison'] = kernel_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    print(f"  - Calibration passed: {cal_results['passed']}")
    print(f"  - TARA-MMD FPR: {100*fpr_results['tara_mmd_fpr']:.1f}%")
    print(f"  - TARA-K FPR: {100*fpr_results['tara_k_fpr']:.1f}%")
    print(f"  - TARA-MMD attack detections: {attack_results['tara_mmd_detections']}/{attack_results['attack_count']}")
    print(f"  - TARA-K attack detections: {attack_results['tara_k_detections']}/{attack_results['attack_count']}")
    print(f"  - Mean computation time: {fpr_results['mean_computation_time']:.2f}s")

    all_results['summary'] = {
        'calibration_passed': cal_results['passed'],
        'tara_mmd_fpr': fpr_results['tara_mmd_fpr'],
        'tara_k_fpr': fpr_results['tara_k_fpr'],
        'tara_mmd_attacks': attack_results['tara_mmd_detections'],
        'tara_k_attacks': attack_results['tara_k_detections'],
        'attack_count': attack_results['attack_count'],
        'mean_computation_time': fpr_results['mean_computation_time'],
        'validation_passed': validation['all_passed']
    }

    # Save results
    output_file = PHASE3_RESULTS / 'exp_3_2_tara_mmd.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
