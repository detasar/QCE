#!/usr/bin/env python3
"""
Experiment 2.3: Detailed Readout Error Analysis

This experiment provides deeper analysis of readout (measurement) errors:
1. Asymmetric readout errors (P(0|1) vs P(1|0))
2. Correlated vs independent readout errors
3. Comparison with gate errors
4. Impact on different measurement bases

Physics Background:
- Readout error: classical bit flip during measurement
- Typically asymmetric: P(0|1) > P(1|0) (decay to ground state)
- Affects ALL measurements including CHSH security samples
- Does NOT affect quantum state, only classical readout

Expected results:
- Asymmetric errors have different effects on Z vs X basis
- Readout error threshold for Bell violation: ~10%
- Readout errors add to QBER directly

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
from utils.circuits import HAS_QISKIT, choose_measurement_settings, create_qkd_circuit
from utils.sifting import sift_keys
from utils.security import calculate_chsh_value, TSIRELSON_BOUND
from utils.qber import estimate_qber

try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError
except ImportError:
    HAS_QISKIT = False


@dataclass
class ReadoutTestResult:
    """Result of QKD test with readout errors."""
    p01: float           # P(1|0) - false positive
    p10: float           # P(0|1) - false negative
    asymmetry: float     # |p01 - p10| / (p01 + p10)

    n_pairs: int
    n_sifted: int
    sifting_rate: float

    chsh_value: float
    chsh_std: float
    chsh_violation: bool

    qber: float
    qber_ci: float

    # Theoretical predictions
    qber_theory: float   # Expected QBER from readout error
    chsh_theory: float   # Expected CHSH degradation

    def to_dict(self) -> dict:
        return {
            'p01': self.p01,
            'p10': self.p10,
            'asymmetry': self.asymmetry,
            'n_pairs': self.n_pairs,
            'n_sifted': self.n_sifted,
            'sifting_rate': self.sifting_rate,
            'chsh_value': self.chsh_value,
            'chsh_std': self.chsh_std,
            'chsh_violation': self.chsh_violation,
            'qber': self.qber,
            'qber_ci': self.qber_ci,
            'qber_theory': self.qber_theory,
            'chsh_theory': self.chsh_theory
        }


def create_readout_noise_model(p01: float, p10: float) -> 'NoiseModel':
    """
    Create noise model with asymmetric readout errors.

    Args:
        p01: P(measure 1 | state 0) - false positive
        p10: P(measure 0 | state 1) - false negative

    Returns:
        NoiseModel with readout errors only
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit not available")

    noise_model = NoiseModel()

    # Readout error matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
    # = [[1-p01, p01], [p10, 1-p10]]
    readout_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
    noise_model.add_all_qubit_readout_error(readout_err)

    return noise_model


def theoretical_qber_from_readout(p01: float, p10: float) -> float:
    """
    Calculate theoretical QBER from readout errors.

    For Bell state |Φ+> with matching bases:
    - Ideal: P(00) = P(11) = 0.5
    - With readout error:
        P(error) = P(00→01) + P(00→10) + P(11→01) + P(11→10)
                 = 0.5*(p01 + p10) + 0.5*(p01 + p10)
                 = p01 + p10 (approximately, for small errors)

    More precisely:
        QBER ≈ p01*(1-p10) + p10*(1-p01) ≈ p01 + p10 for small p
    """
    return p01 * (1 - p10) + p10 * (1 - p01)


def theoretical_chsh_from_readout(p01: float, p10: float) -> float:
    """
    Calculate theoretical CHSH with readout errors.

    Readout error acts as a classical binary symmetric channel on outcomes.
    For error rate p = (p01 + p10)/2:
        Correlator E → (1 - 2p) × E_ideal

    So: S → (1 - 2p) × S_ideal = (1 - p01 - p10) × 2√2
    """
    return (1 - p01 - p10) * TSIRELSON_BOUND


class ReadoutQKDSimulator:
    """QKD simulator with readout errors only."""

    def __init__(self, p01: float, p10: float, seed: int = 42):
        if not HAS_QISKIT:
            raise ImportError("Qiskit not available")

        self.p01 = p01
        self.p10 = p10
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        noise_model = create_readout_noise_model(p01, p10)
        self.backend = AerSimulator(noise_model=noise_model)

    def generate_qkd_data(self, n_pairs: int = 10000,
                           key_fraction: float = 0.90) -> Dict:
        results = {
            'modes': [],
            'alice_bases': [],
            'bob_bases': [],
            'alice_bits': [],
            'bob_bits': [],
            'alice_settings': [],
            'bob_settings': []
        }

        for i in range(n_pairs):
            basis = choose_measurement_settings(key_fraction, self.rng)
            qc = create_qkd_circuit(basis)

            job = self.backend.run(qc, shots=1, seed_simulator=self.seed + i)
            counts = job.result().get_counts()

            outcome = max(counts, key=counts.get)
            bob_bit = int(outcome[0])
            alice_bit = int(outcome[1])

            results['modes'].append(basis.mode.value)
            results['alice_bases'].append(basis.alice_basis)
            results['bob_bases'].append(basis.bob_basis)
            results['alice_bits'].append(alice_bit)
            results['bob_bits'].append(bob_bit)
            results['alice_settings'].append(basis.alice_setting)
            results['bob_settings'].append(basis.bob_setting)

        for key in ['alice_bits', 'bob_bits', 'alice_settings', 'bob_settings']:
            results[key] = np.array(results[key])

        return results


def run_readout_test(p01: float, p10: float, n_pairs: int = 8000,
                     seed: int = 42, verbose: bool = True) -> ReadoutTestResult:
    """
    Run QKD test with specific readout error parameters.

    Args:
        p01: P(1|0) false positive rate
        p10: P(0|1) false negative rate
        n_pairs: Number of Bell pairs
        seed: Random seed
        verbose: Print progress

    Returns:
        ReadoutTestResult with metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Readout Test: P(1|0)={p01:.3f}, P(0|1)={p10:.3f}")
        print("=" * 60)

    # Asymmetry measure
    total = p01 + p10
    asymmetry = abs(p01 - p10) / total if total > 0 else 0

    # Theoretical predictions
    qber_theory = theoretical_qber_from_readout(p01, p10)
    chsh_theory = theoretical_chsh_from_readout(p01, p10)

    if verbose:
        print(f"  Asymmetry: {100*asymmetry:.1f}%")
        print(f"  Theory QBER: {100*qber_theory:.1f}%")
        print(f"  Theory CHSH: {chsh_theory:.3f}")

    # Generate data
    sim = ReadoutQKDSimulator(p01, p10, seed)
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
        qber_result, _, _ = estimate_qber(
            sifting_result.sifted_alice, sifting_result.sifted_bob,
            sample_fraction=0.1, seed=seed + 100
        )
        qber = qber_result.qber
        qber_ci = qber_result.ci_width
    else:
        qber = 0.5
        qber_ci = 0.5

    if verbose:
        print(f"  QBER: {100*qber:.2f}% (theory: {100*qber_theory:.1f}%)")

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
        print(f"  CHSH S: {chsh_value:.3f} (theory: {chsh_theory:.3f})")
        print(f"  Bell violation: {'YES' if chsh_violation else 'NO'}")

    return ReadoutTestResult(
        p01=p01,
        p10=p10,
        asymmetry=asymmetry,
        n_pairs=n_pairs,
        n_sifted=n_sifted,
        sifting_rate=sifting_rate,
        chsh_value=chsh_value,
        chsh_std=chsh_std,
        chsh_violation=chsh_violation,
        qber=qber,
        qber_ci=qber_ci,
        qber_theory=qber_theory,
        chsh_theory=chsh_theory
    )


def run_symmetric_sweep(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Sweep symmetric readout errors (p01 = p10).
    """
    print("\n" + "#" * 60)
    print("SYMMETRIC READOUT ERROR SWEEP")
    print("#" * 60)

    error_rates = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
    results = []

    for p in error_rates:
        result = run_readout_test(p, p, n_pairs=n_pairs,
                                   seed=seed + int(p * 1000))
        results.append({
            'p_symmetric': p,
            'result': result.to_dict()
        })

    # Summary
    print("\n--- Symmetric Readout Error Summary ---")
    print(f"{'p_error':<10} {'QBER':<10} {'QBER_th':<10} {'CHSH':<10} {'CHSH_th':<10} {'Viol':<8}")
    print("-" * 58)

    for r in results:
        viol = "YES" if r['result']['chsh_violation'] else "NO"
        print(f"{r['p_symmetric']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{100*r['result']['qber_theory']:<10.1f}% "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{r['result']['chsh_theory']:<10.3f} "
              f"{viol:<8}")

    return {'type': 'symmetric_sweep', 'results': results}


def run_asymmetric_test(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Test asymmetric readout errors (realistic scenario).

    Typical hardware: P(0|1) > P(1|0) due to relaxation during readout.
    """
    print("\n" + "#" * 60)
    print("ASYMMETRIC READOUT ERROR TEST")
    print("#" * 60)

    configs = [
        {'name': 'Symmetric 5%', 'p01': 0.05, 'p10': 0.05},
        {'name': 'Asymmetric (high p10)', 'p01': 0.02, 'p10': 0.08},
        {'name': 'Asymmetric (high p01)', 'p01': 0.08, 'p10': 0.02},
        {'name': 'Realistic IBM-like', 'p01': 0.01, 'p10': 0.03},
        {'name': 'Realistic IonQ-like', 'p01': 0.005, 'p10': 0.005},
        {'name': 'High asymmetry', 'p01': 0.01, 'p10': 0.10},
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")
        result = run_readout_test(
            config['p01'], config['p10'],
            n_pairs=n_pairs,
            seed=seed + hash(config['name']) % 1000,
            verbose=False
        )

        secure = result.chsh_violation and result.qber < 0.11
        results.append({
            'name': config['name'],
            'config': config,
            'result': result.to_dict(),
            'secure': secure
        })

        status = "SECURE" if secure else "INSECURE"
        print(f"  P(1|0)={config['p01']:.3f}, P(0|1)={config['p10']:.3f}")
        print(f"  CHSH={result.chsh_value:.3f}, QBER={100*result.qber:.1f}% → {status}")

    # Summary
    print("\n--- Asymmetric Readout Error Summary ---")
    print(f"{'Config':<25} {'P(1|0)':<8} {'P(0|1)':<8} {'Asym':<8} {'CHSH':<8} {'QBER':<8} {'Status':<10}")
    print("-" * 75)

    for r in results:
        status = "SECURE" if r['secure'] else "INSECURE"
        print(f"{r['name']:<25} "
              f"{r['config']['p01']:<8.3f} "
              f"{r['config']['p10']:<8.3f} "
              f"{100*r['result']['asymmetry']:<8.0f}% "
              f"{r['result']['chsh_value']:<8.3f} "
              f"{100*r['result']['qber']:<8.1f}% "
              f"{status:<10}")

    return {'type': 'asymmetric_test', 'results': results}


def run_threshold_search(n_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Find readout error threshold for QKD security.

    Search for:
    1. CHSH threshold (S = 2)
    2. QBER threshold (11%)
    """
    print("\n" + "#" * 60)
    print("READOUT ERROR THRESHOLD SEARCH")
    print("#" * 60)

    test_points = np.linspace(0.0, 0.15, 16)
    results = []
    chsh_threshold = None
    qber_threshold = None

    for p in test_points:
        result = run_readout_test(p, p, n_pairs=n_pairs,
                                   seed=seed + int(p * 1000), verbose=False)

        results.append({
            'p_error': p,
            'chsh': result.chsh_value,
            'qber': result.qber,
            'violation': result.chsh_violation
        })

        if chsh_threshold is None and result.chsh_value <= 2.0:
            chsh_threshold = p
        if qber_threshold is None and result.qber >= 0.11:
            qber_threshold = p

    print("\n--- Threshold Search Results ---")
    print(f"{'p_error':<10} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 45)

    for r in results:
        viol = "YES" if r['violation'] else "NO"
        marker = ""
        if r['p_error'] == chsh_threshold:
            marker = " ← CHSH threshold"
        elif r['p_error'] == qber_threshold:
            marker = " ← QBER threshold"
        print(f"{r['p_error']:<10.4f} {r['chsh']:<10.3f} "
              f"{100*r['qber']:<10.1f}% {viol:<10}{marker}")

    print(f"\nCHSH threshold (S=2): p ≈ {chsh_threshold:.4f}" if chsh_threshold else "\nCHSH threshold: Not found")
    print(f"QBER threshold (11%): p ≈ {qber_threshold:.4f}" if qber_threshold else "QBER threshold: Not found")

    return {
        'type': 'threshold_search',
        'results': results,
        'chsh_threshold': chsh_threshold,
        'qber_threshold': qber_threshold
    }


def compare_readout_vs_gate(n_pairs: int = 6000, seed: int = 42) -> dict:
    """
    Compare effect of readout errors vs gate errors.

    Question: Is 5% readout error equivalent to 5% gate error?
    """
    print("\n" + "#" * 60)
    print("READOUT VS GATE ERROR COMPARISON")
    print("#" * 60)

    from utils.circuits import create_noise_model, QKDSimulator

    error_levels = [0.02, 0.05, 0.10]
    results = []

    for p in error_levels:
        print(f"\n--- Error level: {100*p:.0f}% ---")

        # Readout error only
        readout_result = run_readout_test(p, p, n_pairs=n_pairs,
                                           seed=seed + int(p * 1000), verbose=False)

        # Gate error only
        gate_noise = create_noise_model(p_single=p, p_two=2*p, p_readout=0.0)
        gate_sim = QKDSimulator(noise_model=gate_noise, seed=seed + int(p * 1000) + 500)
        gate_data = gate_sim.generate_qkd_data(n_pairs=n_pairs)
        gate_sift = sift_keys(gate_data)

        if gate_sift.n_sifted > 100:
            gate_qber_result, _, _ = estimate_qber(
                gate_sift.sifted_alice, gate_sift.sifted_bob,
                sample_fraction=0.1, seed=seed + 200
            )
            gate_qber = gate_qber_result.qber
        else:
            gate_qber = 0.5

        if len(gate_sift.security_data['alice_outcomes']) >= 50:
            gate_chsh = calculate_chsh_value(gate_sift.security_data)
            gate_chsh_val = gate_chsh.S
            gate_violation = gate_chsh.violation
        else:
            gate_chsh_val = 0
            gate_violation = False

        results.append({
            'error_level': p,
            'readout': {
                'chsh': readout_result.chsh_value,
                'qber': readout_result.qber,
                'violation': readout_result.chsh_violation
            },
            'gate': {
                'chsh': gate_chsh_val,
                'qber': gate_qber,
                'violation': gate_violation
            }
        })

        print(f"  Readout: CHSH={readout_result.chsh_value:.3f}, QBER={100*readout_result.qber:.1f}%")
        print(f"  Gate:    CHSH={gate_chsh_val:.3f}, QBER={100*gate_qber:.1f}%")

    # Summary
    print("\n--- Readout vs Gate Error Comparison ---")
    print(f"{'Error':<8} {'Readout CHSH':<14} {'Gate CHSH':<12} {'Readout QBER':<14} {'Gate QBER':<12}")
    print("-" * 60)

    for r in results:
        print(f"{100*r['error_level']:<8.0f}% "
              f"{r['readout']['chsh']:<14.3f} "
              f"{r['gate']['chsh']:<12.3f} "
              f"{100*r['readout']['qber']:<14.1f}% "
              f"{100*r['gate']['qber']:<12.1f}%")

    return {'type': 'comparison', 'results': results}


def validate_results(all_results: dict) -> dict:
    """Validate experiment results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: Theory matches experiment for symmetric case
    sym = all_results.get('symmetric_sweep', {})
    if sym and 'results' in sym:
        errors = []
        for r in sym['results']:
            if r['result']['qber_theory'] > 0:
                rel_error = abs(r['result']['qber'] - r['result']['qber_theory']) / r['result']['qber_theory']
                errors.append(rel_error)
        mean_error = np.mean(errors) if errors else 0
        theory_ok = mean_error < 0.5  # Within 50% (some statistical noise expected)
        checks.append({
            'name': 'QBER matches theory',
            'passed': theory_ok,
            'detail': f"Mean relative error: {100*mean_error:.1f}%"
        })

    # Check 2: CHSH decreases with readout error
    if sym and 'results' in sym:
        chsh_values = [r['result']['chsh_value'] for r in sym['results']]
        decreasing = sum(1 for i in range(len(chsh_values)-1)
                        if chsh_values[i] >= chsh_values[i+1] - 0.15)
        trend_ok = decreasing >= len(chsh_values) - 2
        checks.append({
            'name': 'CHSH decreases with readout error',
            'passed': trend_ok,
            'detail': f"Decreasing pairs: {decreasing}/{len(chsh_values)-1}"
        })

    # Check 3: Threshold found in reasonable range
    threshold = all_results.get('threshold_search', {})
    if threshold:
        th = threshold.get('chsh_threshold')
        th_ok = th is not None and 0.05 < th < 0.15
        checks.append({
            'name': 'CHSH threshold in expected range',
            'passed': th_ok,
            'detail': f"Found: p={th:.4f}" if th else "Not found"
        })

    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    print(f"\nOverall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete readout error experiment."""
    print("=" * 60)
    print("EXPERIMENT 2.3: DETAILED READOUT ERROR ANALYSIS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Qiskit available: {HAS_QISKIT}")

    if not HAS_QISKIT:
        print("ERROR: Qiskit required")
        return None

    all_results = {
        'experiment': 'exp_2_3_readout_error',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    N_PAIRS = 8000

    # Test 1: Symmetric sweep
    print("\n" + "#" * 60)
    print("TEST 1: SYMMETRIC READOUT ERROR SWEEP")
    print("#" * 60)

    sym_results = run_symmetric_sweep(n_pairs=N_PAIRS, seed=42)
    all_results['tests']['symmetric_sweep'] = sym_results

    # Test 2: Asymmetric test
    print("\n" + "#" * 60)
    print("TEST 2: ASYMMETRIC READOUT ERROR TEST")
    print("#" * 60)

    asym_results = run_asymmetric_test(n_pairs=N_PAIRS, seed=43)
    all_results['tests']['asymmetric_test'] = asym_results

    # Test 3: Threshold search
    print("\n" + "#" * 60)
    print("TEST 3: THRESHOLD SEARCH")
    print("#" * 60)

    threshold_results = run_threshold_search(n_pairs=5000, seed=44)
    all_results['tests']['threshold_search'] = threshold_results

    # Test 4: Readout vs Gate comparison
    print("\n" + "#" * 60)
    print("TEST 4: READOUT VS GATE ERROR COMPARISON")
    print("#" * 60)

    comparison_results = compare_readout_vs_gate(n_pairs=6000, seed=45)
    all_results['tests']['comparison'] = comparison_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    if threshold_results.get('chsh_threshold'):
        print(f"  - CHSH threshold: p ≈ {threshold_results['chsh_threshold']:.4f}")
    if threshold_results.get('qber_threshold'):
        print(f"  - QBER threshold: p ≈ {threshold_results['qber_threshold']:.4f}")

    n_secure = sum(1 for r in asym_results.get('results', []) if r['secure'])
    print(f"  - Secure asymmetric configs: {n_secure}/{len(asym_results.get('results', []))}")

    all_results['summary'] = {
        'overall_passed': validation['all_passed'],
        'chsh_threshold': threshold_results.get('chsh_threshold'),
        'qber_threshold': threshold_results.get('qber_threshold'),
        'n_secure_asymmetric': n_secure
    }

    # Save
    output_file = PHASE2_RESULTS / 'exp_2_3_readout_error.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
