#!/usr/bin/env python3
"""
Experiment 1.2: Key Sifting Protocol

This experiment validates the QKD sifting phase:
1. Basis reconciliation between Alice and Bob
2. Discarding non-matching basis measurements
3. Separation of key bits from security test samples
4. CHSH calculation from security samples

Expected results:
- ~50% sifting rate for uniform basis choice
- 100% correlation for ideal matching bases
- Correct CHSH value from security samples
- Proper separation of key/security data

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
from utils.circuits import IdealBellSimulator, QKDSimulator, HAS_QISKIT
from utils.sifting import (
    sift_keys, calculate_chsh_from_security_data,
    analyze_sifting_statistics, verify_sifted_correlation
)


def run_sifting_test(n_pairs: int = 10000, key_fraction: float = 0.9,
                     visibility: float = 1.0, seed: int = 42) -> dict:
    """
    Run complete sifting protocol test.

    Args:
        n_pairs: Number of Bell pairs
        key_fraction: Fraction for key generation
        visibility: State visibility
        seed: Random seed

    Returns:
        Test results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Sifting Test: visibility={visibility}, n_pairs={n_pairs}")
    print(f"{'='*60}")

    # Generate QKD data
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)

    # Analyze pre-sifting statistics
    pre_stats = analyze_sifting_statistics(qkd_data)

    print(f"\nPre-sifting statistics:")
    print(f"  Total samples: {pre_stats['total_samples']}")
    print(f"  Key samples: {pre_stats['key_samples']} ({100*pre_stats['key_fraction']:.1f}%)")
    print(f"  Security samples: {pre_stats['security_samples']}")
    print(f"\nBasis combinations (key mode):")
    for combo, count in pre_stats['basis_counts'].items():
        pct = 100 * count / pre_stats['key_samples'] if pre_stats['key_samples'] > 0 else 0
        print(f"    {combo}: {count} ({pct:.1f}%)")

    # Perform sifting
    sifting_result = sift_keys(qkd_data)

    print(f"\nSifting results:")
    print(f"  Matching bases: {sifting_result.n_sifted}")
    print(f"  Sifting rate: {100*sifting_result.sifting_rate:.1f}%")
    print(f"  Expected rate: {100*pre_stats['expected_sifting_rate']:.1f}%")

    # Verify sifted correlation
    expected_qber = (1 - visibility) / 2  # For Bell state with visibility v
    correlation_check = verify_sifted_correlation(
        sifting_result,
        expected_qber=expected_qber,
        tolerance=0.03
    )

    print(f"\nCorrelation check:")
    print(f"  Actual QBER: {100*correlation_check['actual_qber']:.2f}%")
    print(f"  Expected QBER: {100*correlation_check['expected_qber']:.2f}%")
    print(f"  Errors: {correlation_check['n_errors']} / {correlation_check['n_sifted']}")
    print(f"  Valid: {'YES' if correlation_check['valid'] else 'NO'}")

    # Calculate CHSH from security samples
    if len(sifting_result.security_data['settings']) > 0:
        chsh_value, correlators, counts = calculate_chsh_from_security_data(
            sifting_result.security_data
        )

        print(f"\nCHSH from security samples:")
        print(f"  S = {chsh_value:.3f}")
        for setting, E in correlators.items():
            print(f"    E{setting} = {E:.3f} (n={counts[setting]})")

        # Expected CHSH for given visibility
        expected_chsh = 2.828 * visibility
        print(f"  Expected S: ~{expected_chsh:.3f}")
        print(f"  Bell violation: {'YES' if chsh_value > 2.0 else 'NO'}")
    else:
        chsh_value = 0
        correlators = {}
        counts = {}

    # Compile results
    results = {
        'params': {
            'n_pairs': n_pairs,
            'key_fraction': key_fraction,
            'visibility': visibility,
            'seed': seed
        },
        'pre_sifting': pre_stats,
        'sifting_result': sifting_result.to_dict(),
        'correlation_check': correlation_check,
        'chsh': {
            'value': chsh_value,
            'correlators': correlators,
            'counts': counts,
            'expected': 2.828 * visibility,
            'bell_violation': chsh_value > 2.0
        }
    }

    return results


def validate_sifting_results(results: dict) -> dict:
    """
    Validate sifting test results.

    Args:
        results: Test results

    Returns:
        Validation report
    """
    validation = {'passed': True, 'checks': []}
    visibility = results['params']['visibility']

    # Check 1: Sifting rate ~50%
    actual_rate = results['sifting_result']['sifting_rate']
    expected_rate = 0.5
    tolerance = 0.05

    check1 = abs(actual_rate - expected_rate) < tolerance
    validation['checks'].append({
        'name': 'sifting_rate',
        'expected': f'{100*expected_rate:.0f}% +/- {100*tolerance:.0f}%',
        'actual': f'{100*actual_rate:.1f}%',
        'passed': check1
    })
    validation['passed'] = validation['passed'] and check1

    # Check 2: Correlation check passed
    check2 = results['correlation_check']['valid']
    validation['checks'].append({
        'name': 'sifted_correlation',
        'expected': f'QBER ~{100*(1-visibility)/2:.1f}%',
        'actual': f'QBER {100*results["correlation_check"]["actual_qber"]:.2f}%',
        'passed': check2
    })
    validation['passed'] = validation['passed'] and check2

    # Check 3: CHSH value
    if visibility > 0.7:
        chsh_val = results['chsh']['value']
        expected_chsh = 2.828 * visibility
        tolerance_chsh = 0.3

        check3 = abs(chsh_val - expected_chsh) < tolerance_chsh
        validation['checks'].append({
            'name': 'chsh_value',
            'expected': f'~{expected_chsh:.2f}',
            'actual': f'{chsh_val:.3f}',
            'passed': check3
        })
        validation['passed'] = validation['passed'] and check3

        # Check 4: Bell violation
        check4 = results['chsh']['bell_violation']
        validation['checks'].append({
            'name': 'bell_violation',
            'expected': 'S > 2.0',
            'actual': f'S = {chsh_val:.3f}',
            'passed': check4
        })
        validation['passed'] = validation['passed'] and check4

    return validation


def run_experiment():
    """Run complete sifting experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.2: KEY SIFTING PROTOCOL")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_2_sifting',
        'timestamp': datetime.now().isoformat(),
        'tests': []
    }

    overall_passed = True

    # Test 1: Ideal visibility
    print("\n" + "=" * 60)
    print("TEST 1: Ideal Visibility (v=1.0)")
    print("=" * 60)

    results1 = run_sifting_test(n_pairs=10000, visibility=1.0, seed=42)
    validation1 = validate_sifting_results(results1)

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

    # Test 2: Noisy channel (v=0.95)
    print("\n" + "=" * 60)
    print("TEST 2: Noisy Channel (v=0.95)")
    print("=" * 60)

    results2 = run_sifting_test(n_pairs=10000, visibility=0.95, seed=43)
    validation2 = validate_sifting_results(results2)

    print("\n--- VALIDATION ---")
    for check in validation2['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation2['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'noisy_v0.95',
        'results': results2,
        'validation': validation2
    })
    overall_passed = overall_passed and validation2['passed']

    # Test 3: Higher noise (v=0.85)
    print("\n" + "=" * 60)
    print("TEST 3: Higher Noise (v=0.85)")
    print("=" * 60)

    results3 = run_sifting_test(n_pairs=10000, visibility=0.85, seed=44)
    validation3 = validate_sifting_results(results3)

    print("\n--- VALIDATION ---")
    for check in validation3['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
    print(f"  Overall: {'PASSED' if validation3['passed'] else 'FAILED'}")

    all_results['tests'].append({
        'name': 'noisy_v0.85',
        'results': results3,
        'validation': validation3
    })
    overall_passed = overall_passed and validation3['passed']

    # Test 4: Qiskit simulation if available
    if HAS_QISKIT:
        print("\n" + "=" * 60)
        print("TEST 4: Qiskit Aer Simulation")
        print("=" * 60)

        print("\nGenerating QKD data with Qiskit Aer...")
        qiskit_sim = QKDSimulator(noise_model=None, seed=45)
        qkd_data = qiskit_sim.generate_qkd_data(n_pairs=1000, key_fraction=0.9)

        pre_stats = analyze_sifting_statistics(qkd_data)
        sifting_result = sift_keys(qkd_data)
        correlation_check = verify_sifted_correlation(sifting_result, expected_qber=0.0, tolerance=0.02)

        print(f"\nSifting rate: {100*sifting_result.sifting_rate:.1f}%")
        print(f"QBER: {100*correlation_check['actual_qber']:.2f}%")

        if len(sifting_result.security_data['settings']) > 0:
            chsh_value, correlators, counts = calculate_chsh_from_security_data(
                sifting_result.security_data
            )
            print(f"CHSH S = {chsh_value:.3f}")
        else:
            chsh_value = 0.0

        results4 = {
            'params': {'n_pairs': 1000, 'key_fraction': 0.9, 'visibility': 1.0, 'seed': 45},
            'pre_sifting': pre_stats,
            'sifting_result': sifting_result.to_dict(),
            'correlation_check': correlation_check,
            'chsh': {
                'value': chsh_value,
                'expected': 2.828,
                'bell_violation': chsh_value > 2.0
            }
        }
        validation4 = validate_sifting_results(results4)

        print("\n--- VALIDATION ---")
        for check in validation4['checks']:
            status = 'PASS' if check['passed'] else 'FAIL'
            print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check['actual']}")
        print(f"  Overall: {'PASSED' if validation4['passed'] else 'FAILED'}")

        all_results['tests'].append({
            'name': 'qiskit_ideal',
            'results': results4,
            'validation': validation4
        })
        overall_passed = overall_passed and validation4['passed']

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    n_tests = len(all_results['tests'])
    n_passed = sum(1 for t in all_results['tests'] if t['validation']['passed'])

    print(f"Tests run: {n_tests}")
    print(f"Tests passed: {n_passed}/{n_tests}")
    print(f"Overall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'tests_run': n_tests,
        'tests_passed': n_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_2_sifting.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
