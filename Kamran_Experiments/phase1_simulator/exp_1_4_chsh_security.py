#!/usr/bin/env python3
"""
Experiment 1.4: CHSH Security Check

This experiment validates CHSH-based security analysis:
1. CHSH value calculation with proper statistics
2. Device-independent security bounds
3. Eve information estimation
4. Key rate from Bell violation
5. Statistical significance testing

Expected results:
- S ≈ 2.828 * visibility
- Violation significance > 5 sigma for good visibility
- Correct Eve information bounds
- Key rate matching theoretical predictions

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
from utils.security import (
    calculate_chsh_value, chsh_security_bound, eve_information_bound,
    analyze_chsh_statistics, expected_chsh_from_visibility,
    TSIRELSON_BOUND, CLASSICAL_BOUND
)


def run_chsh_security_test(visibility: float = 1.0,
                            n_pairs: int = 10000,
                            security_fraction: float = 0.1,
                            seed: int = 42) -> dict:
    """
    Run complete CHSH security analysis.

    Args:
        visibility: Bell state visibility
        n_pairs: Number of Bell pairs
        security_fraction: Fraction for security testing
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"CHSH Security Test: visibility={visibility}")
    print(f"{'='*60}")

    # Generate QKD data
    key_fraction = 1 - security_fraction  # e.g., 0.9 for key, 0.1 for security
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)

    # Sift to get security data
    sifting_result = sift_keys(qkd_data)
    security_data = sifting_result.security_data

    n_security = len(security_data['alice_outcomes'])
    print(f"\nSecurity samples: {n_security}")

    # Calculate CHSH
    chsh_result = calculate_chsh_value(security_data)

    print(f"\nCHSH Results:")
    print(f"  S = {chsh_result.S:.4f} +/- {chsh_result.S_std:.4f}")
    print(f"  Classical bound: {CLASSICAL_BOUND}")
    print(f"  Tsirelson bound: {TSIRELSON_BOUND:.4f}")
    print(f"  Bell violation: {'YES' if chsh_result.violation else 'NO'}")
    print(f"  Significance: {chsh_result.sigma_violation:.1f} sigma")

    print(f"\nCorrelators:")
    for setting, E in chsh_result.correlators.items():
        std = chsh_result.correlator_stds[setting]
        n = chsh_result.counts[setting]
        print(f"  E{setting} = {E:.4f} +/- {std:.4f} (n={n})")

    # Expected vs measured
    expected_S = expected_chsh_from_visibility(visibility)
    print(f"\nExpected S: {expected_S:.4f}")
    deviation = abs(chsh_result.S - expected_S)
    print(f"Deviation: {deviation:.4f} ({100*deviation/expected_S:.1f}%)")

    # Statistical analysis
    stats_analysis = analyze_chsh_statistics(chsh_result)
    print(f"\nStatistical Analysis:")
    print(f"  z-score: {stats_analysis['z_score']:.2f}")
    print(f"  p-value: {stats_analysis['p_value']:.2e}")
    print(f"  95% CI: [{stats_analysis['ci_95'][0]:.3f}, {stats_analysis['ci_95'][1]:.3f}]")
    print(f"  Excludes classical: {'YES' if stats_analysis['excludes_classical'] else 'NO'}")
    print(f"  Tsirelson efficiency: {100*stats_analysis['tsirelson_efficiency']:.1f}%")

    # Security bounds
    security_bounds = chsh_security_bound(chsh_result.S)
    print(f"\nDevice-Independent Security:")
    print(f"  Min-entropy: {security_bounds['min_entropy']:.4f} bits")
    print(f"  Guessing probability: {security_bounds['guessing_probability']:.4f}")
    print(f"  Key rate bound: {security_bounds['key_rate_bound']:.4f}")
    print(f"  Security margin: {security_bounds['security_margin']:.3f} ({security_bounds.get('security_margin_percent', 0):.1f}%)")

    # Eve information
    eve_bounds = eve_information_bound(chsh_result.S)
    print(f"\nEve Information Bound:")
    print(f"  I(A:E) <= {eve_bounds['eve_info_upper']:.4f}")
    print(f"  Secure: {'YES' if eve_bounds['secure'] else 'NO'}")

    # Compile results
    results = {
        'params': {
            'visibility': visibility,
            'n_pairs': n_pairs,
            'security_fraction': security_fraction,
            'seed': seed
        },
        'n_security_samples': n_security,
        'expected_S': expected_S,
        'chsh': chsh_result.to_dict(),
        'statistics': {
            'z_score': stats_analysis['z_score'],
            'p_value': stats_analysis['p_value'],
            'ci_95_lower': stats_analysis['ci_95'][0],
            'ci_95_upper': stats_analysis['ci_95'][1],
            'excludes_classical': stats_analysis['excludes_classical'],
            'tsirelson_efficiency': stats_analysis['tsirelson_efficiency']
        },
        'security_bounds': security_bounds,
        'eve_bounds': eve_bounds
    }

    return results


def validate_chsh_results(results: dict) -> dict:
    """
    Validate CHSH security results.

    Args:
        results: Test results

    Returns:
        Validation report
    """
    validation = {'passed': True, 'checks': []}
    visibility = results['params']['visibility']
    expected_S = results['expected_S']

    # Check 1: CHSH close to expected
    actual_S = results['chsh']['S']
    tolerance = 0.2  # Allow ~7% deviation
    check1 = abs(actual_S - expected_S) < tolerance
    validation['checks'].append({
        'name': 'chsh_accuracy',
        'expected': f'S ≈ {expected_S:.3f}',
        'actual': f'S = {actual_S:.3f}',
        'passed': check1
    })
    validation['passed'] = validation['passed'] and check1

    # Check 2: Bell violation (for v > 0.71)
    if visibility > 0.71:
        check2 = results['chsh']['violation']
        validation['checks'].append({
            'name': 'bell_violation',
            'expected': 'S > 2',
            'actual': f"S = {actual_S:.3f} ({'violated' if check2 else 'not violated'})",
            'passed': check2
        })
        validation['passed'] = validation['passed'] and check2

    # Check 3: Statistical significance (for v > 0.8)
    if visibility > 0.8:
        sigma = results['chsh']['sigma_violation']
        check3 = sigma > 3  # At least 3 sigma
        validation['checks'].append({
            'name': 'statistical_significance',
            'expected': '> 3 sigma',
            'actual': f'{sigma:.1f} sigma',
            'passed': check3
        })
        validation['passed'] = validation['passed'] and check3

    # Check 4: CI excludes classical bound (for v > 0.8)
    if visibility > 0.8:
        check4 = results['statistics']['excludes_classical']
        validation['checks'].append({
            'name': 'ci_excludes_classical',
            'expected': 'CI_lower > 2',
            'actual': f"[{results['statistics']['ci_95_lower']:.3f}, {results['statistics']['ci_95_upper']:.3f}]",
            'passed': check4
        })
        validation['passed'] = validation['passed'] and check4

    # Check 5: Security margin positive (for v > 0.71)
    if visibility > 0.71:
        margin = results['security_bounds']['security_margin']
        check5 = margin > 0
        validation['checks'].append({
            'name': 'positive_security_margin',
            'expected': 'margin > 0',
            'actual': f'margin = {margin:.3f}',
            'passed': check5
        })
        validation['passed'] = validation['passed'] and check5

    return validation


def run_experiment():
    """Run complete CHSH security experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.4: CHSH SECURITY CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_4_chsh_security',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }

    overall_passed = True

    # Test 1: Ideal visibility (S ≈ 2.828)
    print("\n" + "=" * 60)
    print("TEST 1: Ideal Visibility (S ≈ 2.828)")
    print("=" * 60)

    results1 = run_chsh_security_test(visibility=1.0, n_pairs=20000, seed=42)
    validation1 = validate_chsh_results(results1)

    print("\n--- VALIDATION ---")
    for check in validation1['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation1['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'ideal_v1.0',
        'results': results1,
        'validation': validation1
    })
    overall_passed = overall_passed and validation1['passed']

    # Test 2: High visibility (v=0.95, S ≈ 2.69)
    print("\n" + "=" * 60)
    print("TEST 2: High Visibility (v=0.95)")
    print("=" * 60)

    results2 = run_chsh_security_test(visibility=0.95, n_pairs=20000, seed=43)
    validation2 = validate_chsh_results(results2)

    print("\n--- VALIDATION ---")
    for check in validation2['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation2['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'high_v0.95',
        'results': results2,
        'validation': validation2
    })
    overall_passed = overall_passed and validation2['passed']

    # Test 3: Moderate visibility (v=0.85, S ≈ 2.40)
    print("\n" + "=" * 60)
    print("TEST 3: Moderate Visibility (v=0.85)")
    print("=" * 60)

    results3 = run_chsh_security_test(visibility=0.85, n_pairs=20000, seed=44)
    validation3 = validate_chsh_results(results3)

    print("\n--- VALIDATION ---")
    for check in validation3['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation3['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'moderate_v0.85',
        'results': results3,
        'validation': validation3
    })
    overall_passed = overall_passed and validation3['passed']

    # Test 4: Near threshold (v=0.75, S ≈ 2.12)
    print("\n" + "=" * 60)
    print("TEST 4: Near Threshold (v=0.75)")
    print("=" * 60)

    results4 = run_chsh_security_test(visibility=0.75, n_pairs=20000, seed=45)
    validation4 = validate_chsh_results(results4)

    print("\n--- VALIDATION ---")
    for check in validation4['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation4['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'near_threshold_v0.75',
        'results': results4,
        'validation': validation4
    })
    overall_passed = overall_passed and validation4['passed']

    # Test 5: Below threshold (v=0.70, S ≈ 1.98)
    print("\n" + "=" * 60)
    print("TEST 5: Below Threshold (v=0.70)")
    print("=" * 60)

    results5 = run_chsh_security_test(visibility=0.70, n_pairs=20000, seed=46)
    validation5 = validate_chsh_results(results5)

    print("\n--- VALIDATION ---")
    for check in validation5['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation5['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'below_threshold_v0.70',
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
    print("\nCHSH Security Summary:")
    print("-" * 80)
    print(f"{'Visibility':<12} {'Expected S':<12} {'Measured S':<14} {'Sigma':<10} {'Secure':<10}")
    print("-" * 80)
    for test in all_results['tests']:
        v = test['results']['params']['visibility']
        exp_s = test['results']['expected_S']
        meas_s = test['results']['chsh']['S']
        sigma = test['results']['chsh']['sigma_violation']
        secure = 'YES' if test['results']['chsh']['violation'] else 'NO'
        print(f"{v:<12.2f} {exp_s:<12.3f} {meas_s:<14.3f} {sigma:<10.1f} {secure:<10}")

    all_results['summary'] = {
        'tests_run': n_tests,
        'tests_passed': n_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_4_chsh_security.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
