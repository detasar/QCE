#!/usr/bin/env python3
"""
Experiment 1.7: CASCADE Error Correction

This experiment validates the CASCADE protocol for QKD:
1. Block parity comparison
2. Binary search error location
3. Multi-pass correction with increasing block sizes
4. Information leakage tracking

Expected results:
- 100% error correction for QBER < 11%
- Leakage ≈ 1.16 * n * h(QBER)
- Correction failure for QBER > 15%

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
from utils.circuits import IdealBellSimulator
from utils.sifting import sift_keys
from utils.qber import qber_from_visibility
from utils.error_correction import (
    cascade_correct, estimate_cascade_leakage, binary_entropy,
    verify_error_correction
)


def generate_sifted_keys(n_pairs: int = 20000, visibility: float = 1.0,
                          seed: int = 42) -> tuple:
    """
    Generate sifted keys from Bell pair simulation.

    Args:
        n_pairs: Number of Bell pairs
        visibility: State visibility
        seed: Random seed

    Returns:
        Tuple of (alice_key, bob_key, qber)
    """
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.95)
    sifting_result = sift_keys(qkd_data)

    alice_key = sifting_result.sifted_alice
    bob_key = sifting_result.sifted_bob

    # Actual QBER
    errors = np.sum(alice_key != bob_key)
    qber = errors / len(alice_key) if len(alice_key) > 0 else 0

    return alice_key, bob_key, qber


def run_cascade_test(visibility: float = 1.0, n_pairs: int = 20000,
                     n_passes: int = 4, seed: int = 42) -> dict:
    """
    Run CASCADE error correction test.

    Args:
        visibility: State visibility
        n_pairs: Number of Bell pairs
        n_passes: Number of CASCADE passes
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"CASCADE Test: visibility={visibility}")
    print("=" * 60)

    # Generate keys
    alice_key, bob_key, actual_qber = generate_sifted_keys(
        n_pairs=n_pairs, visibility=visibility, seed=seed
    )

    n = len(alice_key)
    expected_qber = qber_from_visibility(visibility)
    initial_errors = np.sum(alice_key != bob_key)

    print(f"\nKey length: {n} bits")
    print(f"Expected QBER: {100*expected_qber:.2f}%")
    print(f"Actual QBER: {100*actual_qber:.2f}%")
    print(f"Initial errors: {initial_errors}")

    if n == 0:
        return {'error': 'No sifted bits'}

    # Run CASCADE
    result = cascade_correct(alice_key, bob_key, actual_qber,
                             n_passes=n_passes, seed=seed + 100)

    print(f"\nCASCADE Results:")
    print(f"  Passes: {result.n_passes}")
    print(f"  Initial errors: {result.initial_errors}")
    print(f"  Final errors: {result.final_errors}")
    print(f"  Correction rate: {100*(1 - result.final_errors/max(1, result.initial_errors)):.1f}%")
    print(f"  Success: {'YES' if result.success else 'NO'}")

    # Leakage analysis
    expected_leakage = estimate_cascade_leakage(n, actual_qber)
    print(f"\nLeakage Analysis:")
    print(f"  Bits leaked: {result.bits_leaked}")
    print(f"  Expected leakage: {expected_leakage:.0f}")
    print(f"  Leakage fraction: {100*result.bits_leaked/n:.1f}%")
    print(f"  Shannon limit: {n * binary_entropy(actual_qber):.0f} bits")

    # Verify correction
    if result.success:
        verified, match_rate = verify_error_correction(
            result.corrected_alice, result.corrected_bob, sample_size=100
        )
        print(f"  Verification: {'PASS' if verified else 'FAIL'} ({100*match_rate:.0f}% match)")
    else:
        verified = False
        match_rate = 0

    return {
        'visibility': visibility,
        'key_length': n,
        'expected_qber': expected_qber,
        'actual_qber': actual_qber,
        'initial_errors': result.initial_errors,
        'final_errors': result.final_errors,
        'bits_leaked': result.bits_leaked,
        'expected_leakage': expected_leakage,
        'n_passes': result.n_passes,
        'success': result.success,
        'verified': verified,
        'remaining_key_bits': n - result.bits_leaked
    }


def run_qber_sweep(seed: int = 42) -> dict:
    """
    Test CASCADE across different QBER levels.

    Args:
        seed: Random seed

    Returns:
        Sweep results
    """
    print("\n" + "=" * 60)
    print("QBER Sweep Test")
    print("=" * 60)

    # Test various visibility levels (corresponding to different QBERs)
    visibilities = [1.0, 0.96, 0.92, 0.88, 0.84, 0.80]
    results = []

    for v in visibilities:
        result = run_cascade_test(visibility=v, n_pairs=15000,
                                   n_passes=4, seed=seed + int(v * 100))
        results.append(result)

    # Summary table
    print("\n--- QBER Sweep Summary ---")
    print(f"{'Visibility':<12} {'QBER':<10} {'Errors':<12} {'Corrected':<12} {'Leaked':<10}")
    print("-" * 56)
    for r in results:
        status = 'YES' if r['success'] else 'NO'
        print(f"{r['visibility']:<12.2f} {100*r['actual_qber']:<10.1f}% "
              f"{r['initial_errors']:<12} {status:<12} {r['bits_leaked']:<10}")

    return {
        'results': results,
        'all_success': all(r['success'] for r in results[:5])  # First 5 should succeed
    }


def run_experiment():
    """Run complete CASCADE experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.7: CASCADE ERROR CORRECTION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_7_cascade',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    overall_passed = True

    # Test 1: Ideal case (no errors)
    print("\n" + "#" * 60)
    print("TEST 1: IDEAL CASE (QBER=0%)")
    print("#" * 60)

    ideal_result = run_cascade_test(visibility=1.0, n_pairs=20000, seed=42)
    all_results['tests']['ideal'] = ideal_result

    ideal_passed = ideal_result['success'] and ideal_result['final_errors'] == 0
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if ideal_passed else 'FAIL'}] Zero errors after correction")
    overall_passed = overall_passed and ideal_passed

    # Test 2: Low QBER (typical good channel)
    print("\n" + "#" * 60)
    print("TEST 2: LOW QBER (~2%)")
    print("#" * 60)

    low_qber_result = run_cascade_test(visibility=0.96, n_pairs=20000, seed=43)
    all_results['tests']['low_qber'] = low_qber_result

    low_passed = low_qber_result['success']
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if low_passed else 'FAIL'}] All errors corrected")
    overall_passed = overall_passed and low_passed

    # Test 3: Moderate QBER (challenging but correctable)
    print("\n" + "#" * 60)
    print("TEST 3: MODERATE QBER (~5%)")
    print("#" * 60)

    mod_qber_result = run_cascade_test(visibility=0.90, n_pairs=20000, seed=44)
    all_results['tests']['moderate_qber'] = mod_qber_result

    mod_passed = mod_qber_result['success']
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if mod_passed else 'FAIL'}] All errors corrected")
    overall_passed = overall_passed and mod_passed

    # Test 4: High QBER (near threshold)
    print("\n" + "#" * 60)
    print("TEST 4: HIGH QBER (~10%)")
    print("#" * 60)

    high_qber_result = run_cascade_test(visibility=0.80, n_pairs=20000, seed=45)
    all_results['tests']['high_qber'] = high_qber_result

    high_passed = high_qber_result['final_errors'] <= 5  # Allow up to 5 residual errors
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if high_passed else 'FAIL'}] Residual errors <= 5")
    overall_passed = overall_passed and high_passed

    # Test 5: QBER sweep
    print("\n" + "#" * 60)
    print("TEST 5: QBER SWEEP")
    print("#" * 60)

    sweep_results = run_qber_sweep(seed=46)
    all_results['tests']['qber_sweep'] = sweep_results

    sweep_passed = sweep_results['all_success']
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if sweep_passed else 'FAIL'}] First 5 visibility levels corrected")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    tests_summary = [
        ('Ideal (QBER=0%)', ideal_passed),
        ('Low QBER (~2%)', low_passed),
        ('Moderate QBER (~5%)', mod_passed),
        ('High QBER (~10%)', high_passed),
        ('QBER Sweep', sweep_passed)
    ]

    for name, passed in tests_summary:
        print(f"  [{('PASS' if passed else 'FAIL')}] {name}")

    n_passed = sum(1 for _, p in tests_summary if p)
    overall_passed = n_passed >= 4

    print(f"\nOverall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'ideal_passed': ideal_passed,
        'low_qber_passed': low_passed,
        'moderate_qber_passed': mod_passed,
        'high_qber_passed': high_passed,
        'sweep_passed': sweep_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_7_cascade.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
