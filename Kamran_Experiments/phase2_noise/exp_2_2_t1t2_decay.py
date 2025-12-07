#!/usr/bin/env python3
"""
Experiment 2.2: T1/T2 Decay Simulation

This experiment analyzes QKD protocol performance under thermal relaxation:
1. T1 decay: amplitude damping (energy loss)
2. T2 decay: dephasing (phase coherence loss)
3. Combined T1/T2 effects on CHSH and QBER

Physics Background:
- T1: Spontaneous emission, |1⟩ → |0⟩ with rate 1/T1
- T2: Pure dephasing, off-diagonal elements decay with rate 1/T2
- T2 ≤ 2*T1 (physical constraint)
- Bell state fidelity: F ≈ exp(-t/T2) for dephasing-dominated

Expected results:
- Short T1/T2 → high QBER, low CHSH
- Bell violation requires T2 >> gate_time
- Asymmetric T1 vs T2 effects on different bases

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

# Check for thermal relaxation
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
    HAS_THERMAL = True
except ImportError:
    HAS_THERMAL = False


@dataclass
class T1T2TestResult:
    """Result of QKD test with T1/T2 decay."""
    T1: float           # T1 in microseconds
    T2: float           # T2 in microseconds
    gate_time_1q: float  # Single-qubit gate time in ns
    gate_time_2q: float  # Two-qubit gate time in ns

    n_pairs: int
    n_sifted: int
    sifting_rate: float

    chsh_value: float
    chsh_std: float
    chsh_violation: bool

    qber: float
    qber_ci: float

    # Derived metrics
    t1_gate_ratio: float    # T1 / total_gate_time
    t2_gate_ratio: float    # T2 / total_gate_time
    coherence_factor: float  # exp(-t_total/T2)

    def to_dict(self) -> dict:
        return {
            'T1_us': self.T1,
            'T2_us': self.T2,
            'gate_time_1q_ns': self.gate_time_1q,
            'gate_time_2q_ns': self.gate_time_2q,
            'n_pairs': self.n_pairs,
            'n_sifted': self.n_sifted,
            'sifting_rate': self.sifting_rate,
            'chsh_value': self.chsh_value,
            'chsh_std': self.chsh_std,
            'chsh_violation': self.chsh_violation,
            'qber': self.qber,
            'qber_ci': self.qber_ci,
            't1_gate_ratio': self.t1_gate_ratio,
            't2_gate_ratio': self.t2_gate_ratio,
            'coherence_factor': self.coherence_factor
        }


def create_t1t2_noise_model(T1_us: float, T2_us: float,
                             gate_time_1q_ns: float = 35.0,
                             gate_time_2q_ns: float = 300.0) -> 'NoiseModel':
    """
    Create noise model with T1/T2 thermal relaxation.

    Args:
        T1_us: T1 relaxation time in microseconds
        T2_us: T2 dephasing time in microseconds
        gate_time_1q_ns: Single-qubit gate time in nanoseconds
        gate_time_2q_ns: Two-qubit gate time in nanoseconds

    Returns:
        NoiseModel with thermal relaxation errors
    """
    if not HAS_THERMAL:
        raise ImportError("Qiskit thermal relaxation not available")

    # Convert to same units (nanoseconds)
    T1_ns = T1_us * 1000.0
    T2_ns = T2_us * 1000.0

    # Validate T2 <= 2*T1 (physical constraint)
    if T2_ns > 2 * T1_ns:
        T2_ns = 2 * T1_ns

    noise_model = NoiseModel()

    # Single-qubit gates
    single_gates = ['h', 'x', 'y', 'z', 'sx', 'ry', 'rz', 'id']
    for gate in single_gates:
        try:
            error = thermal_relaxation_error(T1_ns, T2_ns, gate_time_1q_ns)
            noise_model.add_all_qubit_quantum_error(error, gate)
        except Exception:
            pass  # Skip if gate not supported

    # Two-qubit gates
    two_gates = ['cx', 'cz', 'ecr']
    for gate in two_gates:
        try:
            # For two-qubit gates, create error for each qubit
            error_q0 = thermal_relaxation_error(T1_ns, T2_ns, gate_time_2q_ns)
            error_q1 = thermal_relaxation_error(T1_ns, T2_ns, gate_time_2q_ns)
            combined_error = error_q0.tensor(error_q1)
            noise_model.add_all_qubit_quantum_error(combined_error, gate)
        except Exception:
            pass

    return noise_model


def calculate_circuit_time(gate_time_1q_ns: float, gate_time_2q_ns: float) -> float:
    """
    Calculate total circuit time for QKD Bell measurement.

    Bell circuit: H(q0) + CX(q0,q1) + RY(q0) + RY(q1)
    - 1 H gate (1q)
    - 1 CX gate (2q)
    - 2 RY gates (1q, parallel)

    Returns:
        Total time in nanoseconds
    """
    # H + CX + max(RY, RY) = H + CX + RY
    return gate_time_1q_ns + gate_time_2q_ns + gate_time_1q_ns


class T1T2QKDSimulator:
    """QKD simulator with T1/T2 thermal relaxation noise."""

    def __init__(self, T1_us: float, T2_us: float,
                 gate_time_1q_ns: float = 35.0,
                 gate_time_2q_ns: float = 300.0,
                 seed: int = 42):
        """
        Initialize simulator with T1/T2 parameters.

        Args:
            T1_us: T1 in microseconds
            T2_us: T2 in microseconds
            gate_time_1q_ns: Single-qubit gate time in ns
            gate_time_2q_ns: Two-qubit gate time in ns
            seed: Random seed
        """
        if not HAS_THERMAL:
            raise ImportError("Qiskit not installed")

        self.T1_us = T1_us
        self.T2_us = T2_us
        self.gate_time_1q_ns = gate_time_1q_ns
        self.gate_time_2q_ns = gate_time_2q_ns
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Create noise model
        noise_model = create_t1t2_noise_model(T1_us, T2_us,
                                               gate_time_1q_ns, gate_time_2q_ns)
        self.backend = AerSimulator(noise_model=noise_model)

    def generate_qkd_data(self, n_pairs: int = 10000,
                           key_fraction: float = 0.90) -> Dict:
        """Generate QKD measurement data with T1/T2 noise."""
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


def run_t1t2_test(T1_us: float, T2_us: float,
                  gate_time_1q_ns: float = 35.0,
                  gate_time_2q_ns: float = 300.0,
                  n_pairs: int = 8000, seed: int = 42,
                  verbose: bool = True) -> T1T2TestResult:
    """
    Run QKD test with specific T1/T2 parameters.

    Args:
        T1_us: T1 in microseconds
        T2_us: T2 in microseconds
        gate_time_1q_ns: Single-qubit gate time in ns
        gate_time_2q_ns: Two-qubit gate time in ns
        n_pairs: Number of Bell pairs
        seed: Random seed
        verbose: Print progress

    Returns:
        T1T2TestResult with metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"T1/T2 Test: T1={T1_us} μs, T2={T2_us} μs")
        print("=" * 60)

    # Create simulator and generate data
    sim = T1T2QKDSimulator(T1_us, T2_us, gate_time_1q_ns, gate_time_2q_ns, seed)
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

    # Derived metrics
    total_time_ns = calculate_circuit_time(gate_time_1q_ns, gate_time_2q_ns)
    T1_ns = T1_us * 1000.0
    T2_ns = T2_us * 1000.0

    t1_ratio = T1_ns / total_time_ns
    t2_ratio = T2_ns / total_time_ns
    coherence_factor = np.exp(-total_time_ns / T2_ns)

    if verbose:
        print(f"  T1/gate_time: {t1_ratio:.1f}")
        print(f"  T2/gate_time: {t2_ratio:.1f}")
        print(f"  Coherence factor: {coherence_factor:.4f}")

    return T1T2TestResult(
        T1=T1_us,
        T2=T2_us,
        gate_time_1q=gate_time_1q_ns,
        gate_time_2q=gate_time_2q_ns,
        n_pairs=n_pairs,
        n_sifted=n_sifted,
        sifting_rate=sifting_rate,
        chsh_value=chsh_value,
        chsh_std=chsh_std,
        chsh_violation=chsh_violation,
        qber=qber,
        qber_ci=qber_ci,
        t1_gate_ratio=t1_ratio,
        t2_gate_ratio=t2_ratio,
        coherence_factor=coherence_factor
    )


def run_t1_sweep(T2_us: float = 100.0, n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Sweep T1 values while keeping T2 fixed.

    Args:
        T2_us: Fixed T2 value
        n_pairs: Bell pairs per test
        seed: Base random seed

    Returns:
        Sweep results
    """
    print("\n" + "#" * 60)
    print(f"T1 SWEEP (T2 fixed at {T2_us} μs)")
    print("#" * 60)

    # T1 values to test (must be >= T2/2 for physical validity)
    T1_min = max(T2_us / 2, 10)  # Physical limit
    T1_values = [T1_min, T2_us * 0.75, T2_us, T2_us * 2, T2_us * 5, T2_us * 10]
    T1_values = sorted(set(T1_values))  # Remove duplicates, sort

    results = []

    for T1 in T1_values:
        result = run_t1t2_test(T1, T2_us, n_pairs=n_pairs,
                                seed=seed + int(T1))
        results.append({
            'T1_us': T1,
            'T2_us': T2_us,
            'result': result.to_dict()
        })

    # Summary
    print("\n--- T1 Sweep Summary ---")
    print(f"{'T1 (μs)':<12} {'T1/T2':<10} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 52)

    for r in results:
        viol = "YES" if r['result']['chsh_violation'] else "NO"
        ratio = r['T1_us'] / r['T2_us']
        print(f"{r['T1_us']:<12.1f} {ratio:<10.2f} "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{viol:<10}")

    return {'type': 't1_sweep', 'T2_fixed': T2_us, 'results': results}


def run_t2_sweep(T1_us: float = 200.0, n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Sweep T2 values while keeping T1 fixed.

    Args:
        T1_us: Fixed T1 value
        n_pairs: Bell pairs per test
        seed: Base random seed

    Returns:
        Sweep results
    """
    print("\n" + "#" * 60)
    print(f"T2 SWEEP (T1 fixed at {T1_us} μs)")
    print("#" * 60)

    # T2 values to test (must be <= 2*T1)
    T2_max = 2 * T1_us
    T2_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, min(200.0, T2_max)]
    T2_values = [t for t in T2_values if t <= T2_max]

    results = []

    for T2 in T2_values:
        result = run_t1t2_test(T1_us, T2, n_pairs=n_pairs,
                                seed=seed + int(T2))
        results.append({
            'T1_us': T1_us,
            'T2_us': T2,
            'result': result.to_dict()
        })

    # Summary
    print("\n--- T2 Sweep Summary ---")
    print(f"{'T2 (μs)':<12} {'T2/T1':<10} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 52)

    for r in results:
        viol = "YES" if r['result']['chsh_violation'] else "NO"
        ratio = r['T2_us'] / r['T1_us']
        print(f"{r['T2_us']:<12.1f} {ratio:<10.2f} "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{viol:<10}")

    return {'type': 't2_sweep', 'T1_fixed': T1_us, 'results': results}


def run_hardware_comparison(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Compare different hardware configurations.

    Simulates:
    - Superconducting (IBM-like): T1~200μs, T2~100μs
    - Trapped ion (IonQ-like): T1~seconds, T2~seconds
    - Ideal (infinite T1/T2)
    """
    print("\n" + "#" * 60)
    print("HARDWARE COMPARISON")
    print("#" * 60)

    configs = [
        {
            'name': 'Ideal (infinite T1/T2)',
            'T1_us': 1e6,  # 1 second
            'T2_us': 1e6,
            'gate_1q_ns': 35,
            'gate_2q_ns': 300
        },
        {
            'name': 'Trapped Ion (IonQ-like)',
            'T1_us': 10000,  # 10 ms
            'T2_us': 1000,   # 1 ms
            'gate_1q_ns': 10000,  # 10 μs
            'gate_2q_ns': 200000  # 200 μs
        },
        {
            'name': 'Superconducting Excellent',
            'T1_us': 300,
            'T2_us': 200,
            'gate_1q_ns': 35,
            'gate_2q_ns': 300
        },
        {
            'name': 'Superconducting Good (IBM-like)',
            'T1_us': 150,
            'T2_us': 80,
            'gate_1q_ns': 35,
            'gate_2q_ns': 400
        },
        {
            'name': 'Superconducting Moderate',
            'T1_us': 80,
            'T2_us': 40,
            'gate_1q_ns': 35,
            'gate_2q_ns': 500
        },
        {
            'name': 'Superconducting Noisy',
            'T1_us': 30,
            'T2_us': 15,
            'gate_1q_ns': 35,
            'gate_2q_ns': 600
        }
    ]

    results = []

    for config in configs:
        print(f"\nTesting: {config['name']}")

        result = run_t1t2_test(
            T1_us=config['T1_us'],
            T2_us=config['T2_us'],
            gate_time_1q_ns=config['gate_1q_ns'],
            gate_time_2q_ns=config['gate_2q_ns'],
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
        print(f"  CHSH={result.chsh_value:.3f}, QBER={100*result.qber:.1f}% → {status}")

    # Summary table
    print("\n--- Hardware Comparison Summary ---")
    print(f"{'Configuration':<30} {'T2/gate':<10} {'CHSH S':<10} {'QBER':<10} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = "SECURE" if r['secure'] else "INSECURE"
        t2_ratio = r['result']['t2_gate_ratio']
        print(f"{r['name']:<30} {t2_ratio:<10.0f} "
              f"{r['result']['chsh_value']:<10.3f} "
              f"{100*r['result']['qber']:<10.1f}% "
              f"{status:<10}")

    return {
        'type': 'hardware_comparison',
        'results': results,
        'n_secure': sum(1 for r in results if r['secure'])
    }


def find_t2_threshold(T1_us: float = 200.0, n_pairs: int = 5000, seed: int = 42) -> dict:
    """
    Find minimum T2 required for secure QKD.

    Binary search for T2 threshold where CHSH drops below 2.
    """
    print("\n" + "#" * 60)
    print("T2 THRESHOLD SEARCH")
    print("#" * 60)

    # Coarse search
    test_points = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    test_points = [t for t in test_points if t <= 2 * T1_us]

    results = []
    threshold_t2 = None

    for T2 in test_points:
        result = run_t1t2_test(T1_us, T2, n_pairs=n_pairs,
                                seed=seed + int(T2 * 10), verbose=False)

        results.append({
            'T2_us': T2,
            'chsh': result.chsh_value,
            'qber': result.qber,
            'violation': result.chsh_violation
        })

        if threshold_t2 is None and result.chsh_value >= 2.0:
            threshold_t2 = T2

    print("\n--- T2 Threshold Search ---")
    print(f"{'T2 (μs)':<12} {'CHSH S':<10} {'QBER':<10} {'Violation':<10}")
    print("-" * 45)

    for r in results:
        viol = "YES" if r['violation'] else "NO"
        marker = " ← threshold" if r['T2_us'] == threshold_t2 else ""
        print(f"{r['T2_us']:<12.2f} {r['chsh']:<10.3f} "
              f"{100*r['qber']:<10.1f}% {viol:<10}{marker}")

    if threshold_t2:
        print(f"\nT2 threshold for Bell violation: ~{threshold_t2:.1f} μs")
    else:
        print("\nNo threshold found in tested range")

    return {
        'type': 't2_threshold_search',
        'T1_fixed': T1_us,
        'results': results,
        'threshold_t2': threshold_t2
    }


def validate_results(all_results: dict) -> dict:
    """Validate experiment results for physics consistency."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: CHSH increases with T2
    t2_sweep = all_results.get('t2_sweep', {})
    if t2_sweep and 'results' in t2_sweep:
        chsh_values = [r['result']['chsh_value'] for r in t2_sweep['results']]
        # Allow for statistical noise, check general trend
        increasing = sum(1 for i in range(len(chsh_values)-1)
                        if chsh_values[i+1] >= chsh_values[i] - 0.15)
        trend_ok = increasing >= len(chsh_values) - 2
        checks.append({
            'name': 'CHSH increases with T2',
            'passed': trend_ok,
            'detail': f"Increasing pairs: {increasing}/{len(chsh_values)-1}"
        })

    # Check 2: Hardware comparison sensible
    hw = all_results.get('hardware_comparison', {})
    if hw and 'results' in hw:
        # Ideal should be best
        ideal_chsh = hw['results'][0]['result']['chsh_value']
        others_chsh = [r['result']['chsh_value'] for r in hw['results'][1:]]
        ideal_best = ideal_chsh >= max(others_chsh) - 0.15
        checks.append({
            'name': 'Ideal hardware gives best CHSH',
            'passed': ideal_best,
            'detail': f"Ideal: {ideal_chsh:.3f}, Others max: {max(others_chsh):.3f}"
        })

    # Check 3: Low T2 gives low CHSH
    threshold = all_results.get('t2_threshold_search', {})
    if threshold and 'results' in threshold:
        low_t2_results = [r for r in threshold['results'] if r['T2_us'] <= 1.0]
        if low_t2_results:
            low_t2_chsh = np.mean([r['chsh'] for r in low_t2_results])
            low_t2_bad = low_t2_chsh < 2.5
            checks.append({
                'name': 'Low T2 gives degraded CHSH',
                'passed': low_t2_bad,
                'detail': f"Mean CHSH at T2<=1μs: {low_t2_chsh:.3f}"
            })

    # Print results
    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    print(f"\nOverall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete T1/T2 decay experiment."""
    print("=" * 60)
    print("EXPERIMENT 2.2: T1/T2 DECAY SIMULATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Qiskit thermal relaxation available: {HAS_THERMAL}")

    if not HAS_THERMAL:
        print("ERROR: Qiskit with thermal relaxation required")
        return None

    all_results = {
        'experiment': 'exp_2_2_t1t2_decay',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    N_PAIRS = 8000  # Per test

    # Test 1: T2 sweep (most important for dephasing)
    print("\n" + "#" * 60)
    print("TEST 1: T2 SWEEP")
    print("#" * 60)

    t2_results = run_t2_sweep(T1_us=200.0, n_pairs=N_PAIRS, seed=42)
    all_results['tests']['t2_sweep'] = t2_results

    # Test 2: T1 sweep
    print("\n" + "#" * 60)
    print("TEST 2: T1 SWEEP")
    print("#" * 60)

    t1_results = run_t1_sweep(T2_us=100.0, n_pairs=N_PAIRS, seed=43)
    all_results['tests']['t1_sweep'] = t1_results

    # Test 3: Hardware comparison
    print("\n" + "#" * 60)
    print("TEST 3: HARDWARE COMPARISON")
    print("#" * 60)

    hw_results = run_hardware_comparison(n_pairs=N_PAIRS, seed=44)
    all_results['tests']['hardware_comparison'] = hw_results

    # Test 4: T2 threshold search
    print("\n" + "#" * 60)
    print("TEST 4: T2 THRESHOLD SEARCH")
    print("#" * 60)

    threshold_results = find_t2_threshold(T1_us=200.0, n_pairs=5000, seed=45)
    all_results['tests']['t2_threshold_search'] = threshold_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    if threshold_results.get('threshold_t2'):
        print(f"  - T2 threshold for Bell violation: ~{threshold_results['threshold_t2']:.1f} μs")
    print(f"  - Secure hardware configurations: {hw_results['n_secure']}/{len(hw_results['results'])}")

    all_results['summary'] = {
        'overall_passed': validation['all_passed'],
        't2_threshold': threshold_results.get('threshold_t2'),
        'n_secure_configs': hw_results['n_secure']
    }

    # Save results
    output_file = PHASE2_RESULTS / 'exp_2_2_t1t2_decay.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
