#!/usr/bin/env python3
"""
Experiment 1.3: QBER Estimation with Confidence Intervals

This experiment validates QBER estimation methods:
1. Point estimation from sample
2. Wilson score confidence intervals
3. Sample size requirements
4. Security threshold checking
5. Secrecy capacity calculation

Expected results:
- Correct QBER estimation within CI
- Proper sample size determination
- Accurate security threshold evaluation
- Secrecy capacity matching theoretical values

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE1_RESULTS
from utils.circuits import IdealBellSimulator, HAS_QISKIT
from utils.sifting import sift_keys
from utils.qber import (
    estimate_qber, check_qber_security, required_qber_samples,
    qber_from_visibility, secrecy_capacity_bb84,
    secrecy_capacity_device_independent, QBER_THRESHOLDS
)


def run_qber_estimation_test(visibility: float = 1.0,
                              n_pairs: int = 10000,
                              sample_fraction: float = 0.15,
                              confidence: float = 0.95,
                              seed: int = 42) -> dict:
    """
    Run complete QBER estimation test.

    Args:
        visibility: Bell state visibility
        n_pairs: Number of Bell pairs
        sample_fraction: Fraction for QBER estimation
        confidence: Confidence level
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"QBER Estimation: visibility={visibility}")
    print(f"{'='*60}")

    # Generate QKD data and sift
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
    sifting_result = sift_keys(qkd_data)

    print(f"\nSifted bits available: {sifting_result.n_sifted}")

    # Expected QBER from visibility
    expected_qber = qber_from_visibility(visibility)
    print(f"Expected QBER (from visibility): {100*expected_qber:.2f}%")

    # Estimate QBER
    qber_est, remaining_alice, remaining_bob = estimate_qber(
        sifting_result.sifted_alice,
        sifting_result.sifted_bob,
        sample_fraction=sample_fraction,
        confidence=confidence,
        seed=seed + 100
    )

    print(f"\nQBER Estimation:")
    print(f"  Sample size: {qber_est.n_sample} bits")
    print(f"  Errors found: {qber_est.n_errors}")
    print(f"  QBER estimate: {100*qber_est.qber:.2f}%")
    print(f"  {100*confidence:.0f}% CI: [{100*qber_est.ci_lower:.2f}%, {100*qber_est.ci_upper:.2f}%]")
    print(f"  CI width: +/- {100*qber_est.ci_width:.2f}%")
    print(f"  Remaining bits: {qber_est.remaining_bits}")

    # Check if expected QBER is within CI
    qber_in_ci = qber_est.ci_lower <= expected_qber <= qber_est.ci_upper
    print(f"  Expected in CI: {'YES' if qber_in_ci else 'NO'}")

    # Security check for different protocols
    print(f"\nSecurity Thresholds:")
    security_checks = {}
    for protocol, threshold in QBER_THRESHOLDS.items():
        check = check_qber_security(qber_est, protocol)
        security_checks[protocol] = check
        status = 'SECURE' if check['is_secure'] else 'ABORT'
        print(f"  {protocol}: threshold={100*threshold:.1f}%, [{status}] margin={100*check['margin']:.1f}%")

    # Secrecy capacity
    r_bb84 = secrecy_capacity_bb84(qber_est.qber)
    print(f"\nSecrecy Capacity:")
    print(f"  BB84 r = {r_bb84:.4f} bits/raw-bit")

    # Calculate CHSH from security samples and DI capacity
    from utils.sifting import calculate_chsh_from_security_data
    if len(sifting_result.security_data['settings']) > 0:
        chsh_val, _, _ = calculate_chsh_from_security_data(sifting_result.security_data)
        r_di = secrecy_capacity_device_independent(chsh_val)
        print(f"  DI r = {r_di:.4f} (CHSH S = {chsh_val:.3f})")
    else:
        chsh_val = 0
        r_di = 0

    # Compile results
    results = {
        'params': {
            'visibility': visibility,
            'n_pairs': n_pairs,
            'sample_fraction': sample_fraction,
            'confidence': confidence,
            'seed': seed
        },
        'sifting': {
            'n_sifted': sifting_result.n_sifted,
            'sifting_rate': sifting_result.sifting_rate
        },
        'expected_qber': expected_qber,
        'qber_estimate': qber_est.to_dict(),
        'qber_in_ci': qber_in_ci,
        'security_checks': security_checks,
        'secrecy_capacity': {
            'bb84': r_bb84,
            'device_independent': r_di,
            'chsh_value': chsh_val
        },
        'key_bits_remaining': qber_est.remaining_bits
    }

    return results


def validate_qber_results(results: dict) -> dict:
    """
    Validate QBER estimation results.

    Args:
        results: Test results

    Returns:
        Validation report
    """
    validation = {'passed': True, 'checks': []}
    visibility = results['params']['visibility']
    expected_qber = results['expected_qber']

    # Check 1: Expected QBER is within CI
    check1 = results['qber_in_ci']
    validation['checks'].append({
        'name': 'qber_in_confidence_interval',
        'expected': f'{100*expected_qber:.2f}% in CI',
        'actual': f'{100*results["qber_estimate"]["qber"]:.2f}% [{100*results["qber_estimate"]["ci_lower"]:.2f}%, {100*results["qber_estimate"]["ci_upper"]:.2f}%]',
        'passed': check1
    })
    validation['passed'] = validation['passed'] and check1

    # Check 2: QBER close to expected (within 3%)
    actual_qber = results['qber_estimate']['qber']
    check2 = abs(actual_qber - expected_qber) < 0.03
    validation['checks'].append({
        'name': 'qber_accuracy',
        'expected': f'{100*expected_qber:.2f}% +/- 3%',
        'actual': f'{100*actual_qber:.2f}%',
        'passed': check2
    })
    validation['passed'] = validation['passed'] and check2

    # Check 3: Security correctly determined
    if visibility > 0.78:  # Should be secure (QBER < 11%)
        check3 = results['security_checks']['e91']['is_secure']
        validation['checks'].append({
            'name': 'security_determination',
            'expected': 'SECURE for E91',
            'actual': 'SECURE' if check3 else 'ABORT',
            'passed': check3
        })
        validation['passed'] = validation['passed'] and check3

    # Check 4: Secrecy capacity positive when secure
    if visibility > 0.85:
        check4 = results['secrecy_capacity']['bb84'] > 0
        validation['checks'].append({
            'name': 'positive_secrecy_capacity',
            'expected': 'r > 0',
            'actual': f"r = {results['secrecy_capacity']['bb84']:.4f}",
            'passed': check4
        })
        validation['passed'] = validation['passed'] and check4

    return validation


def test_sample_size_calculation():
    """Test sample size requirements."""
    print("\n" + "=" * 60)
    print("Sample Size Requirements Test")
    print("=" * 60)

    test_cases = [
        (0.01, 0.99),  # 1% precision, 99% confidence
        (0.02, 0.95),  # 2% precision, 95% confidence
        (0.05, 0.90),  # 5% precision, 90% confidence
        (0.03, 0.99),  # 3% precision, 99% confidence
    ]

    results = {}
    for precision, confidence in test_cases:
        n_required = required_qber_samples(precision, confidence)
        print(f"  {100*precision:.0f}% precision, {100*confidence:.0f}% confidence → {n_required:,} samples")
        results[f'p{int(100*precision)}_c{int(100*confidence)}'] = n_required

    return results


def run_experiment():
    """Run complete QBER estimation experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.3: QBER ESTIMATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_3_qber',
        'timestamp': datetime.now().isoformat(),
        'tests': [],
        'sample_size_requirements': {}
    }

    overall_passed = True

    # Test sample size calculation
    all_results['sample_size_requirements'] = test_sample_size_calculation()

    # Test 1: Ideal (v=1.0, QBER=0%)
    print("\n" + "=" * 60)
    print("TEST 1: Ideal Visibility (QBER=0%)")
    print("=" * 60)

    results1 = run_qber_estimation_test(visibility=1.0, seed=42)
    validation1 = validate_qber_results(results1)

    print("\n--- VALIDATION ---")
    for check in validation1['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}")
    print(f"  Overall: {'PASSED' if validation1['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'ideal_v1.0',
        'results': results1,
        'validation': validation1
    })
    overall_passed = overall_passed and validation1['passed']

    # Test 2: Low noise (v=0.96, QBER=2%)
    print("\n" + "=" * 60)
    print("TEST 2: Low Noise (QBER~2%)")
    print("=" * 60)

    results2 = run_qber_estimation_test(visibility=0.96, seed=43)
    validation2 = validate_qber_results(results2)

    print("\n--- VALIDATION ---")
    for check in validation2['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}")
    print(f"  Overall: {'PASSED' if validation2['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'low_noise_v0.96',
        'results': results2,
        'validation': validation2
    })
    overall_passed = overall_passed and validation2['passed']

    # Test 3: Moderate noise (v=0.90, QBER=5%)
    print("\n" + "=" * 60)
    print("TEST 3: Moderate Noise (QBER~5%)")
    print("=" * 60)

    results3 = run_qber_estimation_test(visibility=0.90, seed=44)
    validation3 = validate_qber_results(results3)

    print("\n--- VALIDATION ---")
    for check in validation3['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}")
    print(f"  Overall: {'PASSED' if validation3['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'moderate_noise_v0.90',
        'results': results3,
        'validation': validation3
    })
    overall_passed = overall_passed and validation3['passed']

    # Test 4: High noise (v=0.80, QBER=10%)
    print("\n" + "=" * 60)
    print("TEST 4: High Noise (QBER~10%)")
    print("=" * 60)

    results4 = run_qber_estimation_test(visibility=0.80, seed=45)
    validation4 = validate_qber_results(results4)

    print("\n--- VALIDATION ---")
    for check in validation4['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}")
    print(f"  Overall: {'PASSED' if validation4['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'high_noise_v0.80',
        'results': results4,
        'validation': validation4
    })
    overall_passed = overall_passed and validation4['passed']

    # Test 5: Near threshold (v=0.78, QBER~11%)
    print("\n" + "=" * 60)
    print("TEST 5: Near Threshold (QBER~11%)")
    print("=" * 60)

    results5 = run_qber_estimation_test(visibility=0.78, seed=46)
    validation5 = validate_qber_results(results5)

    print("\n--- VALIDATION ---")
    for check in validation5['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}")
    print(f"  Overall: {'PASSED' if validation5['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'near_threshold_v0.78',
        'results': results5,
        'validation': validation5
    })
    overall_passed = overall_passed and validation5['passed']

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    n_tests = len(all_results['tests'])
    n_passed = sum(1 for t in all_results['tests'] if t['validation']['passed'])

    print(f"Tests run: {n_tests}")
    print(f"Tests passed: {n_passed}/{n_tests}")
    print(f"Overall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    # Print summary table
    print("\nQBER Estimation Summary:")
    print("-" * 60)
    print(f"{'Visibility':<12} {'Expected':<10} {'Measured':<10} {'CI Width':<10} {'Status':<8}")
    print("-" * 60)
    for test in all_results['tests']:
        v = test['results']['params']['visibility']
        exp = 100 * test['results']['expected_qber']
        meas = 100 * test['results']['qber_estimate']['qber']
        ci = 100 * test['results']['qber_estimate']['ci_width']
        status = 'PASS' if test['validation']['passed'] else 'FAIL'
        print(f"{v:<12.2f} {exp:<10.2f}% {meas:<10.2f}% +/-{ci:<6.2f}% {status:<8}")

    all_results['summary'] = {
        'tests_run': n_tests,
        'tests_passed': n_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_3_qber.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
