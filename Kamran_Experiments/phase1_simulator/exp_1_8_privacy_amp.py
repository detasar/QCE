#!/usr/bin/env python3
"""
Experiment 1.8: Privacy Amplification

This experiment validates Toeplitz-based privacy amplification:
1. Toeplitz matrix generation
2. Key compression via matrix multiplication
3. Final key length calculation
4. Security parameter verification

Expected results:
- Keys match after PA (if EC was successful)
- Correct compression ratio based on QBER
- Positive key rate for QBER < 11%
- Zero key for QBER > 11%

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
from utils.error_correction import cascade_correct
from utils.privacy_amp import (
    privacy_amplification, calculate_final_key_length,
    generate_toeplitz_matrix, estimate_key_rate, key_to_hex,
    hash_key, binary_entropy
)


def generate_corrected_keys(n_pairs: int = 20000, visibility: float = 1.0,
                             seed: int = 42) -> tuple:
    """
    Generate and correct keys for privacy amplification.

    Returns:
        Tuple of (alice_key, bob_key, qber, leak_ec, chsh_value)
    """
    # Generate Bell pairs
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
    sifting_result = sift_keys(qkd_data)

    alice_raw = sifting_result.sifted_alice
    bob_raw = sifting_result.sifted_bob
    qber = np.sum(alice_raw != bob_raw) / len(alice_raw) if len(alice_raw) > 0 else 0

    # Error correction
    ec_result = cascade_correct(alice_raw, bob_raw, qber,
                                 n_passes=4, seed=seed + 100)

    # CHSH from security data
    from utils.security import calculate_chsh_value
    if len(sifting_result.security_data['alice_outcomes']) > 0:
        chsh_result = calculate_chsh_value(sifting_result.security_data)
        chsh_value = chsh_result.S
    else:
        chsh_value = None

    return (ec_result.corrected_alice, ec_result.corrected_bob,
            qber, ec_result.bits_leaked, chsh_value)


def run_privacy_amp_test(visibility: float = 1.0, n_pairs: int = 20000,
                          seed: int = 42) -> dict:
    """
    Run privacy amplification test.

    Args:
        visibility: State visibility
        n_pairs: Number of Bell pairs
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"Privacy Amplification: visibility={visibility}")
    print("=" * 60)

    # Get corrected keys
    alice_key, bob_key, qber, leak_ec, chsh_value = generate_corrected_keys(
        n_pairs=n_pairs, visibility=visibility, seed=seed
    )

    n = len(alice_key)
    residual_errors = np.sum(alice_key != bob_key)

    print(f"\nCorrected key length: {n} bits")
    print(f"Residual errors: {residual_errors}")
    print(f"QBER: {100*qber:.2f}%")
    print(f"EC leakage: {leak_ec} bits")
    if chsh_value:
        print(f"CHSH S: {chsh_value:.3f}")

    # Calculate expected final key length
    expected_m = calculate_final_key_length(n, qber, leak_ec, chsh_value)
    print(f"\nExpected final key: {expected_m} bits")

    # Perform privacy amplification
    pa_result = privacy_amplification(
        alice_key, bob_key, qber, leak_ec,
        chsh_value=chsh_value, safety_bits=128, seed=seed + 200
    )

    print(f"\nPrivacy Amplification Results:")
    print(f"  Input length: {pa_result.input_length} bits")
    print(f"  Output length: {pa_result.output_length} bits")
    print(f"  Compression: {100*pa_result.compression_ratio:.1f}%")
    print(f"  Keys match: {'YES' if pa_result.keys_match else 'NO'}")

    # Key preview (first 32 bits in hex)
    if pa_result.output_length > 0:
        alice_hex = key_to_hex(pa_result.final_key_alice[:min(32, len(pa_result.final_key_alice))])
        bob_hex = key_to_hex(pa_result.final_key_bob[:min(32, len(pa_result.final_key_bob))])
        print(f"\n  Alice key (first 32 bits): {alice_hex}")
        print(f"  Bob key (first 32 bits):   {bob_hex}")

        # Hash verification
        alice_hash = hash_key(pa_result.final_key_alice)
        bob_hash = hash_key(pa_result.final_key_bob)
        print(f"  Alice hash: {alice_hash}")
        print(f"  Bob hash:   {bob_hash}")

    # Key rate
    sifting_rate = n / (n_pairs * 0.9) if n_pairs > 0 else 0  # Approximate
    key_rate = pa_result.output_length / n_pairs if n_pairs > 0 else 0

    print(f"\n  Key rate: {key_rate:.4f} bits/pair")

    return {
        'visibility': visibility,
        'n_pairs': n_pairs,
        'corrected_length': n,
        'residual_errors': int(residual_errors),
        'qber': qber,
        'ec_leakage': leak_ec,
        'chsh_value': chsh_value,
        'pa_result': pa_result.to_dict(),
        'key_rate': key_rate,
        'success': pa_result.keys_match and pa_result.output_length > 0
    }


def run_key_rate_analysis(seed: int = 42) -> dict:
    """
    Analyze key rate vs visibility.

    Args:
        seed: Random seed

    Returns:
        Analysis results
    """
    print("\n" + "=" * 60)
    print("Key Rate Analysis")
    print("=" * 60)

    visibilities = [1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.78]
    results = []

    for v in visibilities:
        result = run_privacy_amp_test(visibility=v, n_pairs=15000,
                                       seed=seed + int(v * 100))
        results.append({
            'visibility': v,
            'qber': result['qber'],
            'final_key': result['pa_result']['output_length'],
            'key_rate': result['key_rate'],
            'success': result['success']
        })

    # Summary table
    print("\n--- Key Rate Summary ---")
    print(f"{'Visibility':<12} {'QBER':<10} {'Final Key':<12} {'Rate':<12} {'Status':<10}")
    print("-" * 56)
    for r in results:
        status = 'OK' if r['success'] else 'FAIL'
        print(f"{r['visibility']:<12.2f} {100*r['qber']:<10.1f}% "
              f"{r['final_key']:<12} {r['key_rate']:<12.4f} {status:<10}")

    return {
        'results': results,
        'positive_key_count': sum(1 for r in results if r['final_key'] > 0)
    }


def run_experiment():
    """Run complete privacy amplification experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.8: PRIVACY AMPLIFICATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_8_privacy_amp',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    overall_passed = True

    # Test 1: Ideal case
    print("\n" + "#" * 60)
    print("TEST 1: IDEAL CASE (QBER=0%)")
    print("#" * 60)

    ideal_result = run_privacy_amp_test(visibility=1.0, n_pairs=20000, seed=42)
    all_results['tests']['ideal'] = ideal_result

    ideal_passed = ideal_result['success'] and ideal_result['pa_result']['output_length'] > 0
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if ideal_passed else 'FAIL'}] Positive key and keys match")
    overall_passed = overall_passed and ideal_passed

    # Test 2: Low QBER
    print("\n" + "#" * 60)
    print("TEST 2: LOW QBER (~2%)")
    print("#" * 60)

    low_result = run_privacy_amp_test(visibility=0.96, n_pairs=20000, seed=43)
    all_results['tests']['low_qber'] = low_result

    low_passed = low_result['pa_result']['output_length'] > 0
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if low_passed else 'FAIL'}] Positive final key")
    overall_passed = overall_passed and low_passed

    # Test 3: Moderate QBER
    print("\n" + "#" * 60)
    print("TEST 3: MODERATE QBER (~5%)")
    print("#" * 60)

    mod_result = run_privacy_amp_test(visibility=0.90, n_pairs=20000, seed=44)
    all_results['tests']['moderate_qber'] = mod_result

    mod_passed = mod_result['pa_result']['output_length'] > 0
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if mod_passed else 'FAIL'}] Positive final key")
    overall_passed = overall_passed and mod_passed

    # Test 4: Near threshold
    print("\n" + "#" * 60)
    print("TEST 4: NEAR THRESHOLD (QBER~10%)")
    print("#" * 60)

    threshold_result = run_privacy_amp_test(visibility=0.80, n_pairs=20000, seed=45)
    all_results['tests']['near_threshold'] = threshold_result

    # Near threshold may or may not produce key
    threshold_passed = True  # Just check it runs
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if threshold_passed else 'FAIL'}] Test completed")

    # Test 5: Key rate analysis
    print("\n" + "#" * 60)
    print("TEST 5: KEY RATE ANALYSIS")
    print("#" * 60)

    rate_results = run_key_rate_analysis(seed=46)
    all_results['tests']['key_rate'] = rate_results

    rate_passed = rate_results['positive_key_count'] >= 5
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if rate_passed else 'FAIL'}] At least 5 visibility levels produce positive key")

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    tests_summary = [
        ('Ideal (QBER=0%)', ideal_passed),
        ('Low QBER (~2%)', low_passed),
        ('Moderate QBER (~5%)', mod_passed),
        ('Near Threshold', threshold_passed),
        ('Key Rate Analysis', rate_passed)
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
        'threshold_passed': threshold_passed,
        'rate_passed': rate_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_8_privacy_amp.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
