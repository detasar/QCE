#!/usr/bin/env python3
"""
Experiment 2.4: Eve Attack Simulation

This experiment simulates various eavesdropping attacks on QKD:
1. Intercept-Resend Attack
2. Partial Interception (fraction of qubits)
3. Decorrelation Attack (correlation reduction)
4. Optimal Cloning Attack

Key Questions:
- At what interception rate is Eve detectable?
- How much key info can Eve gain before detection?
- Does CHSH/TARA provide early warning?

Physics Background:
- Intercept-resend: Eve measures in random basis, resends
  - Wrong basis: 50% chance of error → 25% QBER contribution
  - QBER = p_intercept × 0.25
- Cloning attack: Optimal universal cloner gives fidelity 5/6
  - Eve's information bounded by cloning fidelity
- No-cloning theorem: Eve cannot copy without disturbance

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE2_RESULTS
from utils.circuits import HAS_QISKIT, IdealBellSimulator
from utils.sifting import sift_keys
from utils.security import calculate_chsh_value, TSIRELSON_BOUND
from utils.qber import estimate_qber


@dataclass
class AttackResult:
    """Result of QKD under attack."""
    attack_type: str
    attack_strength: float  # Probability or strength parameter

    n_pairs: int
    n_sifted: int
    sifting_rate: float

    chsh_value: float
    chsh_std: float
    chsh_violation: bool

    qber: float
    qber_ci: float

    # Attack-specific metrics
    eve_info_bits: float     # Eve's information (bits per key bit)
    detected: bool           # Attack detected?
    detection_method: str    # How detected

    # Theoretical predictions
    qber_theory: float
    chsh_theory: float
    eve_info_theory: float

    def to_dict(self) -> dict:
        return {
            'attack_type': self.attack_type,
            'attack_strength': self.attack_strength,
            'n_pairs': self.n_pairs,
            'n_sifted': self.n_sifted,
            'sifting_rate': self.sifting_rate,
            'chsh_value': self.chsh_value,
            'chsh_std': self.chsh_std,
            'chsh_violation': self.chsh_violation,
            'qber': self.qber,
            'qber_ci': self.qber_ci,
            'eve_info_bits': self.eve_info_bits,
            'detected': self.detected,
            'detection_method': self.detection_method,
            'qber_theory': self.qber_theory,
            'chsh_theory': self.chsh_theory,
            'eve_info_theory': self.eve_info_theory
        }


def binary_entropy(p: float) -> float:
    """Binary entropy function H(p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


class InterceptResendAttack:
    """
    Intercept-resend attack simulation.

    Eve intercepts qubit, measures in random basis, resends.
    When Eve's basis doesn't match: 50% error rate.
    Total QBER contribution: p_intercept × 0.25

    Eve's information: ~p_intercept bits per intercepted bit
    """

    def __init__(self, intercept_prob: float, seed: int = 42):
        """
        Args:
            intercept_prob: Probability Eve intercepts each qubit
            seed: Random seed
        """
        self.p = intercept_prob
        self.rng = np.random.RandomState(seed)

    def apply_attack(self, alice_bit: int, alice_basis: str,
                     bob_bit: int, bob_basis: str) -> Tuple[int, int]:
        """
        Apply intercept-resend attack to a single measurement.

        Returns:
            (alice_bit, modified_bob_bit)
        """
        if self.rng.random() >= self.p:
            return alice_bit, bob_bit  # No interception

        # Eve intercepts and measures in random basis
        eve_basis = self.rng.choice(['Z', 'X'])

        # For matching bases, we need to check if Eve-Alice match
        # Eve-Alice basis match: no error from interception
        # Eve-Alice basis mismatch: 50% error
        if eve_basis != alice_basis:
            # Eve measured in wrong basis
            if self.rng.random() < 0.5:
                bob_bit = 1 - bob_bit  # Flip the bit

        return alice_bit, bob_bit

    @staticmethod
    def theoretical_qber(p_intercept: float) -> float:
        """Expected QBER from intercept-resend."""
        return p_intercept * 0.25

    @staticmethod
    def theoretical_eve_info(p_intercept: float) -> float:
        """Eve's information per key bit."""
        # Eve learns the bit value when she intercepts
        # But half the time she measures in wrong basis
        return p_intercept * 0.5  # bits per key bit


class DecorrelationAttack:
    """
    Decorrelation attack - reduces quantum correlations.

    Eve adds noise to reduce CHSH value while minimizing QBER impact.
    Models: partial interception, noisy channel injection, etc.
    """

    def __init__(self, decorrelation_strength: float, seed: int = 42):
        """
        Args:
            decorrelation_strength: 0 = no attack, 1 = full decorrelation
            seed: Random seed
        """
        self.strength = decorrelation_strength
        self.rng = np.random.RandomState(seed)

    def apply_attack(self, alice_bit: int, alice_basis: str,
                     bob_bit: int, bob_basis: str) -> Tuple[int, int]:
        """Apply decorrelation attack."""
        if self.rng.random() < self.strength:
            # Randomize Bob's outcome
            bob_bit = self.rng.randint(0, 2)

        return alice_bit, bob_bit

    @staticmethod
    def theoretical_qber(strength: float) -> float:
        """Expected QBER from decorrelation."""
        # Decorrelation makes Bob's outcome random
        # Random outcome: 50% error rate
        return strength * 0.5

    @staticmethod
    def theoretical_chsh(strength: float) -> float:
        """Expected CHSH with decorrelation."""
        # Correlations are diluted
        return (1 - strength) * TSIRELSON_BOUND


class CloningAttack:
    """
    Optimal cloning attack simulation.

    Eve uses universal quantum cloner with fidelity 5/6.
    This is the best Eve can do without being detected immediately.
    """

    def __init__(self, cloning_fidelity: float = 5/6, seed: int = 42):
        """
        Args:
            cloning_fidelity: Fidelity of cloner (5/6 for optimal universal)
            seed: Random seed
        """
        self.fidelity = cloning_fidelity
        self.rng = np.random.RandomState(seed)

    def apply_attack(self, alice_bit: int, alice_basis: str,
                     bob_bit: int, bob_basis: str) -> Tuple[int, int]:
        """Apply cloning attack effect."""
        # Cloning introduces error with probability 1 - fidelity
        if self.rng.random() >= self.fidelity:
            bob_bit = 1 - bob_bit

        return alice_bit, bob_bit

    def theoretical_qber(self) -> float:
        """QBER from cloning attack."""
        return 1 - self.fidelity

    def theoretical_eve_info(self) -> float:
        """Eve's information from cloning."""
        # Eve's clone has same fidelity
        # Her information is bounded by Holevo quantity
        return binary_entropy(self.fidelity)


class AttackedBellSimulator:
    """
    Bell state simulator with eavesdropping attack.
    """

    def __init__(self, attack, visibility: float = 1.0, seed: int = 42):
        """
        Args:
            attack: Attack object (InterceptResend, Decorrelation, etc.)
            visibility: Base state visibility (before attack)
            seed: Random seed
        """
        self.attack = attack
        self.base_sim = IdealBellSimulator(visibility=visibility, seed=seed)
        self.rng = np.random.RandomState(seed + 1000)

    def generate_qkd_data(self, n_pairs: int = 10000,
                           key_fraction: float = 0.90) -> Dict:
        """Generate QKD data with attack applied."""
        # Get base data
        data = self.base_sim.generate_qkd_data(n_pairs, key_fraction)

        # Apply attack to each measurement
        for i in range(len(data['alice_bits'])):
            a_bit = data['alice_bits'][i]
            b_bit = data['bob_bits'][i]
            a_basis = data['alice_bases'][i]
            b_basis = data['bob_bases'][i]

            a_new, b_new = self.attack.apply_attack(a_bit, a_basis, b_bit, b_basis)
            data['bob_bits'][i] = b_new

        return data


def detect_attack(chsh_value: float, qber: float,
                  chsh_threshold: float = 2.0,
                  qber_threshold: float = 0.11) -> Tuple[bool, str]:
    """
    Detect if attack occurred based on CHSH and QBER.

    Returns:
        (detected, method)
    """
    if chsh_value <= chsh_threshold:
        return True, "CHSH violation lost"
    if qber >= qber_threshold:
        return True, "QBER threshold exceeded"
    return False, "Not detected"


def run_intercept_resend_sweep(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Sweep intercept-resend attack strength.
    """
    print("\n" + "#" * 60)
    print("INTERCEPT-RESEND ATTACK SWEEP")
    print("#" * 60)

    intercept_probs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = []

    for p in intercept_probs:
        print(f"\n--- Intercept probability: {100*p:.0f}% ---")

        attack = InterceptResendAttack(p, seed=seed + int(p * 1000))
        sim = AttackedBellSimulator(attack, visibility=1.0, seed=seed + int(p * 1000))

        qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
        sift = sift_keys(qkd_data)

        # QBER
        if sift.n_sifted > 100:
            qber_result, _, _ = estimate_qber(
                sift.sifted_alice, sift.sifted_bob,
                sample_fraction=0.1, seed=seed + 100
            )
            qber = qber_result.qber
            qber_ci = qber_result.ci_width
        else:
            qber = 0.5
            qber_ci = 0.5

        # CHSH
        security_data = sift.security_data
        if len(security_data['alice_outcomes']) >= 50:
            chsh = calculate_chsh_value(security_data)
            chsh_value = chsh.S
            chsh_std = chsh.S_std
            chsh_violation = chsh.violation
        else:
            chsh_value = 0
            chsh_std = 0
            chsh_violation = False

        # Detection
        detected, method = detect_attack(chsh_value, qber)

        # Theoretical predictions
        qber_th = InterceptResendAttack.theoretical_qber(p)
        eve_info_th = InterceptResendAttack.theoretical_eve_info(p)
        chsh_th = (1 - 2 * qber_th) * TSIRELSON_BOUND  # Approximate

        result = AttackResult(
            attack_type='intercept_resend',
            attack_strength=p,
            n_pairs=n_pairs,
            n_sifted=sift.n_sifted,
            sifting_rate=sift.sifting_rate,
            chsh_value=chsh_value,
            chsh_std=chsh_std,
            chsh_violation=chsh_violation,
            qber=qber,
            qber_ci=qber_ci,
            eve_info_bits=eve_info_th,
            detected=detected,
            detection_method=method,
            qber_theory=qber_th,
            chsh_theory=chsh_th,
            eve_info_theory=eve_info_th
        )

        results.append(result.to_dict())

        status = "DETECTED" if detected else "UNDETECTED"
        print(f"  QBER: {100*qber:.1f}% (theory: {100*qber_th:.1f}%)")
        print(f"  CHSH: {chsh_value:.3f}")
        print(f"  Eve info: {eve_info_th:.3f} bits/key bit")
        print(f"  Status: {status} ({method})")

    # Summary
    print("\n--- Intercept-Resend Summary ---")
    print(f"{'p_int':<8} {'QBER':<10} {'CHSH':<10} {'Eve Info':<10} {'Detected':<15}")
    print("-" * 53)

    for r in results:
        det = "YES" if r['detected'] else "NO"
        print(f"{r['attack_strength']:<8.1f} "
              f"{100*r['qber']:<10.1f}% "
              f"{r['chsh_value']:<10.3f} "
              f"{r['eve_info_theory']:<10.3f} "
              f"{det:<15}")

    return {'type': 'intercept_resend_sweep', 'results': results}


def run_decorrelation_sweep(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Sweep decorrelation attack strength.
    """
    print("\n" + "#" * 60)
    print("DECORRELATION ATTACK SWEEP")
    print("#" * 60)

    strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    for s in strengths:
        print(f"\n--- Decorrelation strength: {100*s:.0f}% ---")

        attack = DecorrelationAttack(s, seed=seed + int(s * 1000))
        sim = AttackedBellSimulator(attack, visibility=1.0, seed=seed + int(s * 1000))

        qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
        sift = sift_keys(qkd_data)

        if sift.n_sifted > 100:
            qber_result, _, _ = estimate_qber(
                sift.sifted_alice, sift.sifted_bob,
                sample_fraction=0.1, seed=seed + 100
            )
            qber = qber_result.qber
        else:
            qber = 0.5

        security_data = sift.security_data
        if len(security_data['alice_outcomes']) >= 50:
            chsh = calculate_chsh_value(security_data)
            chsh_value = chsh.S
            chsh_violation = chsh.violation
        else:
            chsh_value = 0
            chsh_violation = False

        detected, method = detect_attack(chsh_value, qber)

        qber_th = DecorrelationAttack.theoretical_qber(s)
        chsh_th = DecorrelationAttack.theoretical_chsh(s)

        results.append({
            'strength': s,
            'qber': qber,
            'qber_theory': qber_th,
            'chsh_value': chsh_value,
            'chsh_theory': chsh_th,
            'chsh_violation': chsh_violation,
            'detected': detected,
            'detection_method': method
        })

        status = "DETECTED" if detected else "UNDETECTED"
        print(f"  QBER: {100*qber:.1f}% (theory: {100*qber_th:.1f}%)")
        print(f"  CHSH: {chsh_value:.3f} (theory: {chsh_th:.3f})")
        print(f"  Status: {status}")

    print("\n--- Decorrelation Summary ---")
    print(f"{'Strength':<10} {'QBER':<10} {'CHSH':<10} {'CHSH_th':<10} {'Detected':<10}")
    print("-" * 50)

    for r in results:
        det = "YES" if r['detected'] else "NO"
        print(f"{r['strength']:<10.1f} "
              f"{100*r['qber']:<10.1f}% "
              f"{r['chsh_value']:<10.3f} "
              f"{r['chsh_theory']:<10.3f} "
              f"{det:<10}")

    return {'type': 'decorrelation_sweep', 'results': results}


def run_cloning_attack_test(n_pairs: int = 8000, seed: int = 42) -> dict:
    """
    Test optimal cloning attack.
    """
    print("\n" + "#" * 60)
    print("OPTIMAL CLONING ATTACK TEST")
    print("#" * 60)

    fidelities = [1.0, 5/6, 0.75, 0.6, 0.5]
    results = []

    for f in fidelities:
        print(f"\n--- Cloning fidelity: {f:.3f} ---")

        attack = CloningAttack(f, seed=seed + int(f * 1000))
        sim = AttackedBellSimulator(attack, visibility=1.0, seed=seed + int(f * 1000))

        qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
        sift = sift_keys(qkd_data)

        if sift.n_sifted > 100:
            qber_result, _, _ = estimate_qber(
                sift.sifted_alice, sift.sifted_bob,
                sample_fraction=0.1, seed=seed + 100
            )
            qber = qber_result.qber
        else:
            qber = 0.5

        security_data = sift.security_data
        if len(security_data['alice_outcomes']) >= 50:
            chsh = calculate_chsh_value(security_data)
            chsh_value = chsh.S
            chsh_violation = chsh.violation
        else:
            chsh_value = 0
            chsh_violation = False

        detected, method = detect_attack(chsh_value, qber)

        qber_th = 1 - f
        eve_info = binary_entropy(f)

        results.append({
            'fidelity': f,
            'qber': qber,
            'qber_theory': qber_th,
            'chsh_value': chsh_value,
            'chsh_violation': chsh_violation,
            'eve_info': eve_info,
            'detected': detected,
            'detection_method': method
        })

        status = "DETECTED" if detected else "UNDETECTED"
        print(f"  QBER: {100*qber:.1f}% (theory: {100*qber_th:.1f}%)")
        print(f"  CHSH: {chsh_value:.3f}")
        print(f"  Eve info: {eve_info:.3f} bits")
        print(f"  Status: {status}")

    print("\n--- Cloning Attack Summary ---")
    print(f"{'Fidelity':<10} {'QBER':<10} {'CHSH':<10} {'Eve Info':<10} {'Detected':<10}")
    print("-" * 50)

    for r in results:
        det = "YES" if r['detected'] else "NO"
        print(f"{r['fidelity']:<10.3f} "
              f"{100*r['qber']:<10.1f}% "
              f"{r['chsh_value']:<10.3f} "
              f"{r['eve_info']:<10.3f} "
              f"{det:<10}")

    return {'type': 'cloning_attack', 'results': results}


def find_detection_threshold(n_pairs: int = 15000, seed: int = 42, n_runs: int = 3) -> dict:
    """
    Find minimum attack strength for detection.
    Uses multiple runs for statistical robustness.
    """
    print("\n" + "#" * 60)
    print("DETECTION THRESHOLD SEARCH")
    print("#" * 60)
    print(f"(n_pairs={n_pairs}, n_runs={n_runs} per point)")

    # Fine-grained search for intercept-resend
    test_probs = np.linspace(0.0, 0.6, 13)
    results = []
    detection_threshold = None

    for p in test_probs:
        # Multiple runs for averaging
        chsh_values = []
        qber_values = []
        detected_count = 0

        for run in range(n_runs):
            run_seed = seed + int(p * 1000) + run * 100
            attack = InterceptResendAttack(p, seed=run_seed)
            sim = AttackedBellSimulator(attack, visibility=1.0, seed=run_seed)

            qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=0.9)
            sift = sift_keys(qkd_data)

            if sift.n_sifted > 100:
                qber_result, _, _ = estimate_qber(
                    sift.sifted_alice, sift.sifted_bob,
                    sample_fraction=0.1, seed=run_seed + 100
                )
                qber = qber_result.qber
            else:
                qber = 0.5

            security_data = sift.security_data
            if len(security_data['alice_outcomes']) >= 50:
                chsh = calculate_chsh_value(security_data)
                chsh_value = chsh.S
            else:
                chsh_value = 0

            chsh_values.append(chsh_value)
            qber_values.append(qber)

            detected, method = detect_attack(chsh_value, qber)
            if detected:
                detected_count += 1

        # Average results
        avg_chsh = np.mean(chsh_values)
        avg_qber = np.mean(qber_values)
        std_chsh = np.std(chsh_values)

        # Detection by majority vote (detected if majority of runs detected)
        detected = detected_count >= (n_runs + 1) // 2
        eve_info = InterceptResendAttack.theoretical_eve_info(p)

        results.append({
            'p_intercept': p,
            'qber': avg_qber,
            'qber_std': np.std(qber_values),
            'chsh_value': avg_chsh,
            'chsh_std': std_chsh,
            'eve_info': eve_info,
            'detected': detected,
            'detection_rate': detected_count / n_runs,
            'method': f"{detected_count}/{n_runs} runs detected"
        })

        if detection_threshold is None and detected:
            detection_threshold = p

    print("\n--- Detection Threshold Search ---")
    print(f"{'p_int':<8} {'QBER':<10} {'CHSH':<10} {'Eve Info':<10} {'Detected':<12}")
    print("-" * 50)

    for r in results:
        det = "YES" if r['detected'] else "NO"
        marker = " ← threshold" if r['p_intercept'] == detection_threshold else ""
        print(f"{r['p_intercept']:<8.2f} "
              f"{100*r['qber']:<10.1f}% "
              f"{r['chsh_value']:<10.3f} "
              f"{r['eve_info']:<10.3f} "
              f"{det:<12}{marker}")

    if detection_threshold:
        eve_info_at_threshold = InterceptResendAttack.theoretical_eve_info(detection_threshold)
        print(f"\nDetection threshold: p_intercept ≈ {detection_threshold:.2f}")
        print(f"Eve's info at threshold: {eve_info_at_threshold:.3f} bits/key bit")
    else:
        print("\nNo detection in tested range")

    return {
        'type': 'detection_threshold',
        'results': results,
        'threshold': detection_threshold
    }


def validate_results(all_results: dict) -> dict:
    """Validate attack simulation results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: Zero attack = no detection
    ir = all_results.get('intercept_resend_sweep', {})
    if ir and 'results' in ir:
        zero_attack = [r for r in ir['results'] if r['attack_strength'] == 0.0]
        if zero_attack:
            no_detect_at_zero = not zero_attack[0]['detected']
            checks.append({
                'name': 'No attack = not detected',
                'passed': no_detect_at_zero,
                'detail': f"Zero attack detected: {zero_attack[0]['detected']}"
            })

    # Check 2: Full attack = detected
    if ir and 'results' in ir:
        full_attack = [r for r in ir['results'] if r['attack_strength'] == 1.0]
        if full_attack:
            detect_at_full = full_attack[0]['detected']
            checks.append({
                'name': 'Full attack = detected',
                'passed': detect_at_full,
                'detail': f"Full attack QBER: {100*full_attack[0]['qber']:.1f}%"
            })

    # Check 3: QBER increases with attack strength (monotonic trend)
    if ir and 'results' in ir:
        qbers = [r['qber'] for r in ir['results']]
        # Allow 2% tolerance for statistical noise
        increasing = sum(1 for i in range(len(qbers)-1)
                        if qbers[i+1] >= qbers[i] - 0.02)
        trend_ok = increasing >= len(qbers) - 2
        checks.append({
            'name': 'QBER increases with attack',
            'passed': trend_ok,
            'detail': f"Increasing pairs: {increasing}/{len(qbers)-1}"
        })

    # Check 3b: CHSH decreases with attack strength (monotonic trend)
    if ir and 'results' in ir:
        chshs = [r['chsh_value'] for r in ir['results']]
        # Allow 0.1 tolerance for statistical noise
        decreasing = sum(1 for i in range(len(chshs)-1)
                        if chshs[i+1] <= chshs[i] + 0.1)
        chsh_trend_ok = decreasing >= len(chshs) - 2
        checks.append({
            'name': 'CHSH decreases with attack',
            'passed': chsh_trend_ok,
            'detail': f"Decreasing pairs: {decreasing}/{len(chshs)-1}"
        })

    # Check 4: Optimal cloner (5/6 fidelity) gives ~17% QBER
    cloning = all_results.get('cloning_attack', {})
    if cloning and 'results' in cloning:
        optimal = [r for r in cloning['results'] if abs(r['fidelity'] - 5/6) < 0.01]
        if optimal:
            qber_ok = abs(optimal[0]['qber'] - (1 - 5/6)) < 0.05
            checks.append({
                'name': 'Optimal cloner gives ~17% QBER',
                'passed': qber_ok,
                'detail': f"QBER: {100*optimal[0]['qber']:.1f}% (expected: 16.7%)"
            })

    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    print(f"\nOverall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete Eve attack simulation experiment."""
    print("=" * 60)
    print("EXPERIMENT 2.4: EVE ATTACK SIMULATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_2_4_eve_attack',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    N_PAIRS = 12000  # Increased for statistical robustness

    # Test 1: Intercept-Resend Sweep
    print("\n" + "#" * 60)
    print("TEST 1: INTERCEPT-RESEND ATTACK")
    print("#" * 60)

    ir_results = run_intercept_resend_sweep(n_pairs=N_PAIRS, seed=42)
    all_results['tests']['intercept_resend_sweep'] = ir_results

    # Test 2: Decorrelation Sweep
    print("\n" + "#" * 60)
    print("TEST 2: DECORRELATION ATTACK")
    print("#" * 60)

    decor_results = run_decorrelation_sweep(n_pairs=N_PAIRS, seed=43)
    all_results['tests']['decorrelation_sweep'] = decor_results

    # Test 3: Cloning Attack
    print("\n" + "#" * 60)
    print("TEST 3: CLONING ATTACK")
    print("#" * 60)

    cloning_results = run_cloning_attack_test(n_pairs=N_PAIRS, seed=44)
    all_results['tests']['cloning_attack'] = cloning_results

    # Test 4: Detection Threshold
    print("\n" + "#" * 60)
    print("TEST 4: DETECTION THRESHOLD")
    print("#" * 60)

    threshold_results = find_detection_threshold(n_pairs=5000, seed=45)
    all_results['tests']['detection_threshold'] = threshold_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Security Findings:")
    if threshold_results.get('threshold'):
        th = threshold_results['threshold']
        eve_info = InterceptResendAttack.theoretical_eve_info(th)
        print(f"  - Detection threshold: p_intercept ≈ {th:.2f}")
        print(f"  - Eve's max undetected info: {eve_info:.3f} bits/key bit")
        print(f"  - Security margin: {100*(1 - eve_info):.0f}%")

    # Count detections
    n_ir_detected = sum(1 for r in ir_results.get('results', []) if r['detected'])
    n_ir_total = len(ir_results.get('results', []))
    print(f"  - Intercept-resend detected: {n_ir_detected}/{n_ir_total}")

    all_results['summary'] = {
        'overall_passed': validation['all_passed'],
        'detection_threshold': threshold_results.get('threshold'),
        'n_detected': n_ir_detected,
        'n_total': n_ir_total
    }

    # Save
    output_file = PHASE2_RESULTS / 'exp_2_4_eve_attack.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
