#!/usr/bin/env python3
"""
Experiment 1.5: TARA-k Integration

This experiment validates TARA-k (KS-based) conformal prediction detector
for QKD security monitoring:
1. Calibration on honest quantum data
2. Detection of eavesdropper attacks
3. False positive rate verification
4. Attack detection AUC evaluation

TARA-k tests if p-value distribution deviates from Uniform(0,1):
- Under H0 (honest): p-values are uniform
- Under H1 (Eve): p-values deviate, KS statistic increases

Expected results:
- Low KS statistic for honest data (~0.05)
- High KS statistic for attack data (>0.2)
- Type I error < 5% at threshold 0.2
- Detection of intercept-resend and PNS attacks

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE1_RESULTS
from utils.circuits import IdealBellSimulator
from utils.sifting import sift_keys
from utils.tara_integration import TARAk, TARAk_Correlation, qkd_to_tara_format, create_eve_data


def generate_tara_data(n_pairs: int = 10000, visibility: float = 1.0,
                        security_fraction: float = 0.5, seed: int = 42) -> dict:
    """
    Generate data for TARA experiments.

    Uses high security fraction to get enough samples for calibration.

    Args:
        n_pairs: Number of Bell pairs
        visibility: State visibility
        security_fraction: Fraction for security testing
        seed: Random seed

    Returns:
        TARA-formatted data
    """
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    key_fraction = 1 - security_fraction
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)
    sifting_result = sift_keys(qkd_data)

    return qkd_to_tara_format(sifting_result.security_data)


def run_tara_k_calibration_test(seed: int = 42) -> dict:
    """
    Test TARA-k calibration and baseline behavior.

    Args:
        seed: Random seed

    Returns:
        Test results
    """
    print("\n" + "=" * 60)
    print("TARA-k Calibration Test")
    print("=" * 60)

    # Generate calibration data
    print("\nGenerating calibration data...")
    cal_data = generate_tara_data(n_pairs=10000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)
    n_cal = len(cal_data['x'])
    print(f"  Calibration samples: {n_cal}")

    # Split into train/test for validation
    split = n_cal // 2
    cal_train = {k: v[:split] for k, v in cal_data.items()}
    cal_test = {k: v[split:] for k, v in cal_data.items()}

    # Create detector
    detector = TARAk(cal_train)
    print(f"  Training samples: {split}")
    print(f"  Test samples: {len(cal_test['x'])}")

    # Test on honest data
    result = detector.test(cal_test, threshold=0.2)

    print(f"\nHonest data test:")
    print(f"  KS statistic: {result.statistic:.4f}")
    print(f"  Threshold: {result.threshold}")
    print(f"  Detected: {'YES' if result.detected else 'NO'}")
    print(f"  Mean p-value: {result.mean_p_value:.4f}")

    # Check p-value uniformity
    ks_stat, ks_pval = stats.kstest(result.p_values, 'uniform')
    print(f"\nP-value uniformity test:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  KS p-value: {ks_pval:.4f}")
    print(f"  Uniform: {'YES' if ks_pval > 0.05 else 'NO'}")

    return {
        'n_calibration': split,
        'n_test': len(cal_test['x']),
        'ks_statistic': result.statistic,
        'threshold': result.threshold,
        'detected': result.detected,
        'mean_p_value': result.mean_p_value,
        'p_value_ks_stat': ks_stat,
        'p_value_ks_pval': ks_pval,
        'uniform_pvalues': ks_pval > 0.05
    }


def run_attack_detection_test(attack_type: str, noise_level: float,
                               seed: int = 42) -> dict:
    """
    Test TARA-k attack detection.

    Args:
        attack_type: Type of attack
        noise_level: Attack strength
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"Attack Detection: {attack_type} (noise={noise_level})")
    print("=" * 60)

    # Generate calibration data (honest)
    cal_data = generate_tara_data(n_pairs=10000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Generate test data (honest for baseline)
    honest_test = generate_tara_data(n_pairs=5000, visibility=1.0,
                                      security_fraction=0.5, seed=seed + 100)

    # Create attack data
    eve_test = create_eve_data(honest_test, attack_type=attack_type,
                                noise_level=noise_level, seed=seed + 200)

    # Create detector
    detector = TARAk(cal_data)

    # Test honest
    honest_result = detector.test(honest_test, threshold=0.2)
    print(f"\nHonest data:")
    print(f"  KS statistic: {honest_result.statistic:.4f}")
    print(f"  Detected: {'YES (FALSE POSITIVE)' if honest_result.detected else 'NO (Correct)'}")

    # Test attack
    eve_result = detector.test(eve_test, threshold=0.2)
    print(f"\nAttack data ({attack_type}):")
    print(f"  KS statistic: {eve_result.statistic:.4f}")
    print(f"  Detected: {'YES (Correct)' if eve_result.detected else 'NO (MISSED)'}")

    # Discrimination
    discrimination = eve_result.statistic - honest_result.statistic
    print(f"\nDiscrimination:")
    print(f"  Delta KS: {discrimination:.4f}")
    print(f"  Separable: {'YES' if discrimination > 0.1 else 'NO'}")

    return {
        'attack_type': attack_type,
        'noise_level': noise_level,
        'honest_ks': honest_result.statistic,
        'honest_detected': honest_result.detected,
        'eve_ks': eve_result.statistic,
        'eve_detected': eve_result.detected,
        'discrimination': discrimination,
        'attack_detected_correctly': eve_result.detected and not honest_result.detected
    }


def run_correlation_detector_test(seed: int = 42) -> dict:
    """
    Test the correlation-based TARA detector.

    This detector uses CHSH-like correlations instead of
    individual conditional probabilities.
    """
    print("\nGenerating calibration and test data...")

    # Generate calibration data (larger for stable correlations)
    cal_data = generate_tara_data(n_pairs=20000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Generate honest test data
    honest_test = generate_tara_data(n_pairs=10000, visibility=1.0,
                                      security_fraction=0.5, seed=seed + 100)

    # Generate attack data with reduced visibility (simulates Eve)
    attack_data = generate_tara_data(n_pairs=10000, visibility=0.7,
                                      security_fraction=0.5, seed=seed + 200)

    print(f"  Calibration samples: {len(cal_data['x'])}")
    print(f"  Test samples: {len(honest_test['x'])}")

    # Create detector
    detector = TARAk_Correlation(cal_data, window_size=100)

    # Test honest
    honest_result = detector.test(honest_test)
    print(f"\nHonest test (v=1.0):")
    print(f"  Mean correlation: {np.mean(detector._compute_window_correlations(honest_test)):.3f}")
    print(f"  KS statistic: {honest_result.statistic:.4f}")
    print(f"  Detected: {'YES' if honest_result.detected else 'NO'}")

    # Test attack
    attack_result = detector.test(attack_data)
    print(f"\nAttack test (v=0.7 - simulates Eve):")
    print(f"  Mean correlation: {np.mean(detector._compute_window_correlations(attack_data)):.3f}")
    print(f"  KS statistic: {attack_result.statistic:.4f}")
    print(f"  Detected: {'YES' if attack_result.detected else 'NO'}")

    # Discrimination
    cal_mean = np.mean(detector.cal_correlations)
    honest_mean = np.mean(detector._compute_window_correlations(honest_test))
    attack_mean = np.mean(detector._compute_window_correlations(attack_data))

    print(f"\nCorrelation comparison:")
    print(f"  Calibration mean: {cal_mean:.3f}")
    print(f"  Honest test mean: {honest_mean:.3f}")
    print(f"  Attack test mean: {attack_mean:.3f}")
    print(f"  Drop due to attack: {cal_mean - attack_mean:.3f}")

    success = attack_result.detected and not honest_result.detected
    print(f"\nTest result: {'SUCCESS' if success else 'NEEDS TUNING'}")

    return {
        'calibration_correlation': float(cal_mean),
        'honest_correlation': float(honest_mean),
        'attack_correlation': float(attack_mean),
        'honest_detected': honest_result.detected,
        'attack_detected': attack_result.detected,
        'success': success
    }


def run_false_positive_rate_test(n_trials: int = 50, seed: int = 42) -> dict:
    """
    Estimate false positive rate of TARA-k.

    Args:
        n_trials: Number of trials
        seed: Random seed

    Returns:
        False positive rate statistics
    """
    print("\n" + "=" * 60)
    print(f"False Positive Rate Test (n={n_trials})")
    print("=" * 60)

    false_positives = 0
    ks_stats = []

    for i in range(n_trials):
        trial_seed = seed + i * 1000

        # Generate fresh calibration data
        cal_data = generate_tara_data(n_pairs=5000, visibility=1.0,
                                       security_fraction=0.5, seed=trial_seed)

        # Generate fresh test data (independent)
        test_data = generate_tara_data(n_pairs=2500, visibility=1.0,
                                        security_fraction=0.5, seed=trial_seed + 500)

        detector = TARAk(cal_data)
        result = detector.test(test_data, threshold=0.2)

        ks_stats.append(result.statistic)
        if result.detected:
            false_positives += 1

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_trials} trials...")

    fpr = false_positives / n_trials
    print(f"\nResults:")
    print(f"  False positives: {false_positives}/{n_trials}")
    print(f"  False positive rate: {100*fpr:.1f}%")
    print(f"  Mean KS statistic: {np.mean(ks_stats):.4f}")
    print(f"  Max KS statistic: {np.max(ks_stats):.4f}")

    return {
        'n_trials': n_trials,
        'false_positives': false_positives,
        'false_positive_rate': fpr,
        'mean_ks': float(np.mean(ks_stats)),
        'std_ks': float(np.std(ks_stats)),
        'max_ks': float(np.max(ks_stats)),
        'acceptable': fpr < 0.1  # Should be < 10%
    }


def run_experiment():
    """Run complete TARA-k experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.5: TARA-k INTEGRATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_5_tara_k',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    overall_passed = True

    # Test 1: Calibration
    print("\n" + "#" * 60)
    print("TEST 1: CALIBRATION")
    print("#" * 60)

    cal_results = run_tara_k_calibration_test(seed=42)
    all_results['tests']['calibration'] = cal_results

    cal_passed = not cal_results['detected'] and cal_results['uniform_pvalues']
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if not cal_results['detected'] else 'FAIL'}] No false detection")
    print(f"  [{'PASS' if cal_results['uniform_pvalues'] else 'FAIL'}] Uniform p-values")
    print(f"  Overall: {'PASSED' if cal_passed else 'FAILED'}")
    overall_passed = overall_passed and cal_passed

    # Test 2: Attack detection
    print("\n" + "#" * 60)
    print("TEST 2: ATTACK DETECTION")
    print("#" * 60)

    attack_tests = [
        ('intercept_resend', 0.30),  # Strong attack
        ('intercept_resend', 0.20),  # Medium attack
        ('decorrelation', 0.25),     # Pure decorrelation
        ('pns', 0.30),               # PNS attack
    ]

    attack_results = []
    for attack_type, noise_level in attack_tests:
        result = run_attack_detection_test(attack_type, noise_level, seed=42)
        attack_results.append(result)

    all_results['tests']['attacks'] = attack_results

    # Count detections
    n_detected = sum(1 for r in attack_results if r['eve_detected'])
    n_false_pos = sum(1 for r in attack_results if r['honest_detected'])

    print(f"\n--- ATTACK SUMMARY ---")
    print(f"  Attacks detected: {n_detected}/{len(attack_results)}")
    print(f"  False positives: {n_false_pos}/{len(attack_results)}")

    # Test 2b: Correlation-based detector
    print("\n" + "-" * 40)
    print("CORRELATION-BASED DETECTOR TEST")
    print("-" * 40)

    corr_results = run_correlation_detector_test(seed=42)
    all_results['tests']['correlation_detector'] = corr_results

    # Test 3: False positive rate
    print("\n" + "#" * 60)
    print("TEST 3: FALSE POSITIVE RATE")
    print("#" * 60)

    fpr_results = run_false_positive_rate_test(n_trials=20, seed=42)
    all_results['tests']['false_positive_rate'] = fpr_results

    fpr_passed = fpr_results['acceptable']
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if fpr_passed else 'FAIL'}] FPR < 10%: {100*fpr_results['false_positive_rate']:.1f}%")
    overall_passed = overall_passed and fpr_passed

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Count all passed tests
    corr_passed = corr_results.get('success', False)
    tests_summary = [
        ('Calibration', cal_passed),
        ('Basic Attack Detection', n_detected >= 2),  # At least 2 of 4 detected
        ('Correlation Detector', corr_passed),
        ('False Positive Rate', fpr_passed)
    ]

    for name, passed in tests_summary:
        print(f"  [{('PASS' if passed else 'FAIL')}] {name}")

    n_passed = sum(1 for _, p in tests_summary if p)
    overall_passed = n_passed == len(tests_summary)

    print(f"\nOverall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'calibration_passed': cal_passed,
        'attacks_detected': n_detected,
        'attacks_total': len(attack_results),
        'correlation_detector_passed': corr_passed,
        'fpr': fpr_results['false_positive_rate'],
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_5_tara_k.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
