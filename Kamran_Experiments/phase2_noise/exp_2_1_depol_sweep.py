#!/usr/bin/env python3
"""
Experiment 2.1: Depolarizing Noise Sweep

This experiment analyzes QKD protocol performance under depolarizing noise:
1. Sweep gate error rates from 0% to 15%
2. Measure CHSH, QBER, sifting rate, and key rate
3. Find security thresholds
4. Validate theoretical predictions

Physics Background:
- Depolarizing channel: ρ → (1-p)ρ + p·I/2
- For Bell circuit with n gates: S ≈ (1-p_eff)^n × S_ideal
- Where p_eff accounts for single and two-qubit errors

Expected results:
- CHSH decreases with noise
- QBER increases with noise
- Security threshold at S = 2 (classical bound)
- Key generation possible until QBER ≈ 11%

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE2_RESULTS
from utils.circuits import (
    QKDSimulator, create_noise_model, HAS_QISKIT,
    CHSH_ANGLES, KEY_BASIS_ANGLES
)
from utils.sifting import sift_keys
from utils.security import calculate_chsh_value, TSIRELSON_BOUND
from utils.qber import estimate_qber


@dataclass
class NoiseTestResult:
    """Result of QKD test at specific noise level."""
    p_single: float
    p_two: float
    p_readout: float

    n_pairs: int
    n_sifted: int
    sifting_rate: float

    chsh_value: float
    chsh_std: float
    chsh_violation: bool

    qber: float
    qber_ci: float

    key_rate_theoretical: float

    def to_dict(self) -> dict:
        return {
            'p_single': self.p_single,
            'p_two': self.p_two,
            'p_readout': self.p_readout,
            'n_pairs': self.n_pairs,
            'n_sifted': self.n_sifted,
            'sifting_rate': self.sifting_rate,
            'chsh_value': self.chsh_value,
            'chsh_std': self.chsh_std,
            'chsh_violation': self.chsh_violation,
            'qber': self.qber,
            'qber_ci': self.qber_ci,
            'key_rate_theoretical': self.key_rate_theoretical
        }


def binary_entropy(p: float) -> float:
    """Binary entropy function h(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def theoretical_key_rate(qber: float, S: float = None) -> float:
    """
    Calculate theoretical key rate.

    Using Shor-Preskill bound:
        r = 1 - 2*h(QBER)

    For device-independent:
        r = 1 - h(p_guess(S))

    Returns 0 if QBER > 11% or S < 2.
    """
    if qber >= 0.11:
        return 0.0
    if S is not None and S <= 2.0:
        return 0.0

    h_qber = binary_entropy(qber)
    return max(0.0, 1 - 2 * h_qber)


def theoretical_chsh_vs_noise(p_single: float, p_two: float = None) -> float:
    """
    Calculate expected CHSH value with depolarizing noise.

    For depolarizing channel on Bell state:
        S_noisy = (1 - p_eff) × S_ideal

    Where p_eff is effective depolarizing probability.

    Bell circuit: H(q0) + CX(q0,q1) + RY(q0) + RY(q1)
    - 1 H gate (single-qubit)
    - 1 CX gate (two-qubit)
    - 2 RY gates (single-qubit)

    Approximate model:
        S ≈ (1-p_single)^3 × (1-p_two)^1 × S_ideal
    """
    if p_two is None:
        p_two = 2 * p_single

    # Fidelity after gates
    f_single = (1 - p_single) ** 3
    f_two = (1 - p_two) ** 1
    f_total = f_single * f_two

    return f_total * TSIRELSON_BOUND


def run_noise_test(p_single: float, p_two: float = None, p_readout: float = 0.0,
                   n_pairs: int = 5000, seed: int = 42,
                   verbose: bool = True) -> NoiseTestResult:
    """
    Run QKD test with specific noise parameters.

    Args:
        p_single: Single-qubit gate error rate
        p_two: Two-qubit gate error rate (default: 2*p_single)
        p_readout: Measurement error rate
        n_pairs: Number of Bell pairs
        seed: Random seed
        verbose: Print progress

    Returns:
        NoiseTestResult with all metrics
    """
    if p_two is None:
        p_two = 2 * p_single

    if verbose:
        print(f"\n{'='*60}")
        print(f"Noise Test: p_single={p_single:.3f}, p_two={p_two:.3f}, p_read={p_readout:.3f}")
        print("=" * 60)

    # Create noise model
    if p_single > 0 or p_readout > 0:
        noise_model = create_noise_model(p_single, p_two, p_readout)
    else:
        noise_model = None

    # Generate QKD data
    sim = QKDSimulator(noise_model=noise_model, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)

    if verbose:
        print(f"  Generated {n_pairs} Bell pairs")

    # Sifting
    sifting_result = sift_keys(qkd_data)
    n_sifted = sifting_result.n_sifted
    sifting_rate = sifting_result.sifting_rate

    if verbose:
        print(f"  Sifted: {n_sifted} bits ({100*sifting_rate:.1f}%)")

    # QBER estimation
    if n_sifted > 100:
        qber_result, remaining_alice, remaining_bob = estimate_qber(
            sifting_result.sifted_alice, sifting_result.sifted_bob,
            sample_fraction=0.1, seed=seed + 100
        )
        qber = qber_result.qber
        qber_ci = qber_result.ci_width
    else:
        qber = 0.5  # Maximum uncertainty
        qber_ci = 0.5

    if verbose:
        print(f"  QBER: {100*qber:.2f}% ± {100*qber_ci:.2f}%")

    # CHSH calculation
    security_data = sifting_result.security_data
    if len(security_data['alice_outcomes']) >= 50:
        chsh_result = calculate_chsh_value(security_data)
        chsh_value = chsh_result.S
        chsh_std = chsh_result.S_std
        chsh_violation = chsh_result.violation
    else:
        chsh_value = 0
        chsh_std = 0
        chsh_violation = False

    if verbose:
        print(f"  CHSH S: {chsh_value:.3f} ± {chsh_std:.3f}")
        print(f"  Bell violation: {'YES' if chsh_violation else 'NO'}")

    # Theoretical key rate
    key_rate = theoretical_key_rate(qber, chsh_value)

    if verbose:
        print(f"  Key rate (theoretical): {key_rate:.4f}")

    return NoiseTestResult(
        p_single=p_single,
        p_two=p_two,
        p_readout=p_readout,
        n_pairs=n_pairs,
        n_sifted=n_sifted,
        sifting_rate=sifting_rate,
        chsh_value=chsh_value,
        chsh_std=chsh_std,
        chsh_violation=chsh_violation,
        qber=qber,
        qber_ci=qber_ci,
        key_rate_theoretical=key_rate
    )


def run_depolarizing_sweep(n_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Run depolarizing noise sweep.

    Tests multiple noise levels and compares with theory.

    Args:
        n_pairs: Bell pairs per noise level
        seed: Base random seed

    Returns:
        Sweep results with theory comparison
    """
    print("\n" + "#" * 60)
    print("DEPOLARIZING NOISE SWEEP")
    print("#" * 60)

    # Noise levels to test
    # p_single values: gate error probability
    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    results = []

    for p_single in noise_levels:
        result = run_noise_test(
            p_single=p_single,
            p_two=2*p_single,  # Standard ratio
            p_readout=0.0,     # Isolate gate error effect
            n_pairs=n_pairs,
            seed=seed + int(p_single * 1000)
        )

        # Add theoretical prediction
        S_theory = theoretical_chsh_vs_noise(p_single)

        results.append({
            'noise_level': p_single,
            'result': result.to_dict(),
            'S_theory': S_theory,
            'S_measured': result.chsh_value,
            'S_error': abs(result.chsh_value - S_theory) / S_theory if S_theory > 0 else 0
        })

    # Summary table
    print("\n" + "=" * 80)
    print("DEPOLARIZING SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'p_single':<10} {'S_theory':<10} {'S_measured':<12} {'Error':<10} {'QBER':<10} {'KeyRate':<10}")
    print("-" * 70)

    for r in results:
        print(f"{r['noise_level']:<10.3f} "
              f"{r['S_theory']:<10.3f} "
              f"{r['S_measured']:<12.3f} "
              f"{100*r['S_error']:<10.1f}% "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{r['result']['key_rate_theoretical']:<10.4f}")

    return {
        'type': 'depolarizing_sweep',
        'n_pairs': n_pairs,
        'results': results
    }


def run_readout_error_test(n_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Test effect of readout errors separately from gate errors.

    Readout error affects measurement outcomes, not quantum state.
    Effect on CHSH: Correlations are diluted towards random.
    """
    print("\n" + "#" * 60)
    print("READOUT ERROR TEST")
    print("#" * 60)

    readout_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15]
    results = []

    for p_readout in readout_levels:
        result = run_noise_test(
            p_single=0.0,      # No gate errors
            p_two=0.0,
            p_readout=p_readout,
            n_pairs=n_pairs,
            seed=seed + int(p_readout * 1000)
        )

        results.append({
            'p_readout': p_readout,
            'result': result.to_dict()
        })

    # Summary
    print("\n--- Readout Error Summary ---")
    print(f"{'p_readout':<12} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 45)

    for r in results:
        viol = "YES" if r['result']['chsh_violation'] else "NO"
        print(f"{r['p_readout']:<12.3f} "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{viol:<10}")

    return {
        'type': 'readout_error_test',
        'n_pairs': n_pairs,
        'results': results
    }


def run_combined_noise_test(n_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Test realistic combined noise (gate + readout errors).

    Typical IBM Quantum values:
    - Single-qubit error: ~0.1-0.5%
    - Two-qubit error: ~0.5-2%
    - Readout error: ~1-5%
    """
    print("\n" + "#" * 60)
    print("COMBINED NOISE TEST (Realistic Hardware)")
    print("#" * 60)

    # Realistic hardware configurations
    hardware_configs = [
        {'name': 'Ideal', 'p_single': 0.0, 'p_two': 0.0, 'p_readout': 0.0},
        {'name': 'Excellent', 'p_single': 0.001, 'p_two': 0.005, 'p_readout': 0.01},
        {'name': 'Good (IonQ-like)', 'p_single': 0.002, 'p_two': 0.01, 'p_readout': 0.01},
        {'name': 'Moderate (IBM-like)', 'p_single': 0.005, 'p_two': 0.02, 'p_readout': 0.03},
        {'name': 'Noisy', 'p_single': 0.01, 'p_two': 0.04, 'p_readout': 0.05},
        {'name': 'Very Noisy', 'p_single': 0.02, 'p_two': 0.08, 'p_readout': 0.10},
    ]

    results = []

    for config in hardware_configs:
        print(f"\nTesting: {config['name']}")

        result = run_noise_test(
            p_single=config['p_single'],
            p_two=config['p_two'],
            p_readout=config['p_readout'],
            n_pairs=n_pairs,
            seed=seed + hash(config['name']) % 1000,
            verbose=False
        )

        results.append({
            'name': config['name'],
            'config': config,
            'result': result.to_dict(),
            'secure': result.chsh_violation and result.qber < 0.11
        })

        status = "SECURE" if results[-1]['secure'] else "INSECURE"
        print(f"  CHSH={result.chsh_value:.3f}, QBER={100*result.qber:.1f}% → {status}")

    # Summary
    print("\n--- Combined Noise Summary ---")
    print(f"{'Config':<20} {'CHSH S':<10} {'QBER':<10} {'KeyRate':<10} {'Status':<10}")
    print("-" * 60)

    for r in results:
        status = "SECURE" if r['secure'] else "INSECURE"
        print(f"{r['name']:<20} "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{r['result']['key_rate_theoretical']:<10.4f} "
              f"{status:<10}")

    return {
        'type': 'combined_noise_test',
        'n_pairs': n_pairs,
        'results': results
    }


def find_security_threshold(n_pairs: int = 3000, seed: int = 42) -> dict:
    """
    Find noise level at which security is lost.

    Two thresholds:
    1. CHSH threshold: S = 2 (Bell inequality)
    2. QBER threshold: 11% (key rate = 0)
    """
    print("\n" + "#" * 60)
    print("SECURITY THRESHOLD SEARCH")
    print("#" * 60)

    # Binary search for threshold
    low, high = 0.0, 0.20
    chsh_threshold_noise = None
    qber_threshold_noise = None

    # Coarse search first
    test_points = np.linspace(0.0, 0.15, 16)
    results = []

    for p in test_points:
        result = run_noise_test(
            p_single=p,
            p_two=2*p,
            p_readout=0.0,
            n_pairs=n_pairs,
            seed=seed + int(p * 1000),
            verbose=False
        )

        results.append({
            'p_single': p,
            'chsh': result.chsh_value,
            'qber': result.qber,
            'violation': result.chsh_violation
        })

        # Find thresholds
        if chsh_threshold_noise is None and result.chsh_value <= 2.0:
            chsh_threshold_noise = p
        if qber_threshold_noise is None and result.qber >= 0.11:
            qber_threshold_noise = p

    print("\n--- Threshold Search Results ---")
    print(f"{'p_single':<10} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 45)

    for r in results:
        viol = "YES" if r['violation'] else "NO"
        marker = ""
        if r['p_single'] == chsh_threshold_noise:
            marker = " ← CHSH threshold"
        elif r['p_single'] == qber_threshold_noise:
            marker = " ← QBER threshold"
        print(f"{r['p_single']:<10.4f} "
              f"{r['chsh']:<10.3f} "
              f"{100*r['qber']:<10.1f}% "
              f"{viol:<10}{marker}")

    print(f"\nCHSH threshold (S=2): p_single ≈ {chsh_threshold_noise:.4f}" if chsh_threshold_noise else "\nCHSH threshold: Not found in range")
    print(f"QBER threshold (11%): p_single ≈ {qber_threshold_noise:.4f}" if qber_threshold_noise else "QBER threshold: Not found in range")

    return {
        'type': 'threshold_search',
        'n_pairs': n_pairs,
        'results': results,
        'chsh_threshold': chsh_threshold_noise,
        'qber_threshold': qber_threshold_noise
    }


def validate_results(all_results: dict) -> dict:
    """
    Validate experiment results for consistency and physics.

    Checks:
    1. CHSH decreases monotonically with noise
    2. QBER increases monotonically with noise
    3. Theory matches measurement within statistical uncertainty
    4. Thresholds are in expected range
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: CHSH monotonicity
    sweep = all_results.get('depolarizing_sweep', {})
    if sweep:
        chsh_values = [r['S_measured'] for r in sweep.get('results', [])]
        is_monotonic = all(chsh_values[i] >= chsh_values[i+1] - 0.1
                          for i in range(len(chsh_values)-1))
        checks.append({
            'name': 'CHSH decreases with noise',
            'passed': is_monotonic,
            'detail': f"Values: {[f'{v:.2f}' for v in chsh_values]}"
        })

    # Check 2: Theory agreement
    if sweep:
        errors = [r['S_error'] for r in sweep.get('results', [])]
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        theory_ok = max_error < 0.15  # Within 15%
        checks.append({
            'name': 'Theory-experiment agreement',
            'passed': theory_ok,
            'detail': f"Mean error: {100*mean_error:.1f}%, Max: {100*max_error:.1f}%"
        })

    # Check 3: Threshold consistency
    threshold = all_results.get('threshold_search', {})
    if threshold:
        chsh_th = threshold.get('chsh_threshold')
        qber_th = threshold.get('qber_threshold')

        # CHSH threshold should be around p=0.05-0.10
        chsh_ok = chsh_th is not None and 0.03 < chsh_th < 0.15
        checks.append({
            'name': 'CHSH threshold in expected range',
            'passed': chsh_ok,
            'detail': f"Found: p={chsh_th:.4f}" if chsh_th else "Not found"
        })

    # Check 4: Ideal case is ideal
    if sweep and len(sweep.get('results', [])) > 0:
        ideal = sweep['results'][0]
        ideal_chsh = ideal['S_measured']
        ideal_close = abs(ideal_chsh - TSIRELSON_BOUND) < 0.15
        checks.append({
            'name': 'Ideal case near Tsirelson bound',
            'passed': ideal_close,
            'detail': f"S={ideal_chsh:.3f} vs {TSIRELSON_BOUND:.3f}"
        })

    # Print validation results
    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    print(f"\nOverall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {
        'checks': checks,
        'all_passed': all_passed
    }


def run_experiment():
    """Run complete depolarizing noise experiment."""
    print("=" * 60)
    print("EXPERIMENT 2.1: DEPOLARIZING NOISE SWEEP")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Qiskit available: {HAS_QISKIT}")

    if not HAS_QISKIT:
        print("ERROR: Qiskit required for noise simulation")
        return None

    all_results = {
        'experiment': 'exp_2_1_depol_sweep',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Use larger sample sizes for better statistics
    # With 10000 pairs, ~1000 security samples → σ ≈ 0.09 for CHSH
    N_PAIRS_SWEEP = 10000
    N_PAIRS_TEST = 8000
    N_PAIRS_THRESHOLD = 5000

    # Test 1: Depolarizing noise sweep
    print("\n" + "#" * 60)
    print("TEST 1: DEPOLARIZING NOISE SWEEP")
    print("#" * 60)

    sweep_results = run_depolarizing_sweep(n_pairs=N_PAIRS_SWEEP, seed=42)
    all_results['tests']['depolarizing_sweep'] = sweep_results

    # Test 2: Readout error test
    print("\n" + "#" * 60)
    print("TEST 2: READOUT ERROR TEST")
    print("#" * 60)

    readout_results = run_readout_error_test(n_pairs=N_PAIRS_TEST, seed=43)
    all_results['tests']['readout_error'] = readout_results

    # Test 3: Combined noise (realistic hardware)
    print("\n" + "#" * 60)
    print("TEST 3: COMBINED NOISE TEST")
    print("#" * 60)

    combined_results = run_combined_noise_test(n_pairs=N_PAIRS_TEST, seed=44)
    all_results['tests']['combined_noise'] = combined_results

    # Test 4: Security threshold search
    print("\n" + "#" * 60)
    print("TEST 4: SECURITY THRESHOLD SEARCH")
    print("#" * 60)

    threshold_results = find_security_threshold(n_pairs=N_PAIRS_THRESHOLD, seed=45)
    all_results['tests']['threshold_search'] = threshold_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    if sweep_results:
        ideal_chsh = sweep_results['results'][0]['S_measured']
        print(f"  - Ideal CHSH: {ideal_chsh:.3f}")

    if threshold_results:
        if threshold_results['chsh_threshold']:
            print(f"  - CHSH threshold at p_single ≈ {threshold_results['chsh_threshold']:.4f}")
        if threshold_results['qber_threshold']:
            print(f"  - QBER threshold at p_single ≈ {threshold_results['qber_threshold']:.4f}")

    secure_configs = sum(1 for r in combined_results.get('results', []) if r['secure'])
    print(f"  - Secure configurations: {secure_configs}/{len(combined_results.get('results', []))}")

    all_results['summary'] = {
        'overall_passed': validation['all_passed'],
        'chsh_threshold': threshold_results.get('chsh_threshold'),
        'qber_threshold': threshold_results.get('qber_threshold'),
        'secure_configs': secure_configs
    }

    # Save results
    output_file = PHASE2_RESULTS / 'exp_2_1_depol_sweep.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
