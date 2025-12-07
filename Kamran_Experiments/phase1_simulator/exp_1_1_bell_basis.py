#!/usr/bin/env python3
"""
Experiment 1.1: Bell Pair + Random Basis Circuit

This experiment validates the core quantum circuit components:
1. Bell state preparation (|Phi+> = (|00> + |11>)/sqrt(2))
2. Random basis selection (key vs security mode)
3. Measurement statistics verification

Expected results:
- Perfect correlation for matching bases
- 50% correlation for non-matching bases
- CHSH value close to 2.828 (Tsirelson bound)
- Correct key/security sample ratio

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE1_RESULTS
from utils.circuits import (
    IdealBellSimulator, QKDSimulator, HAS_QISKIT,
    create_noise_model, MeasurementMode
)


def analyze_qkd_data(data: dict) -> dict:
    """
    Comprehensive analysis of QKD measurement data.

    Args:
        data: Dictionary with measurement results

    Returns:
        Dictionary with analysis results
    """
    n_total = len(data['modes'])

    # Separate by mode
    key_mask = np.array([m == 'key' for m in data['modes']])
    sec_mask = np.array([m == 'security' for m in data['modes']])

    n_key = np.sum(key_mask)
    n_security = np.sum(sec_mask)

    results = {
        'total_pairs': n_total,
        'key_samples': int(n_key),
        'security_samples': int(n_security),
        'key_fraction': n_key / n_total if n_total > 0 else 0
    }

    # Analyze key samples
    alice_bases = np.array(data['alice_bases'])
    bob_bases = np.array(data['bob_bases'])
    alice_bits = data['alice_bits']
    bob_bits = data['bob_bits']

    if n_key > 0:
        # Matching vs non-matching bases
        matching = alice_bases[key_mask] == bob_bases[key_mask]
        n_matching = np.sum(matching)
        n_non_matching = np.sum(~matching)

        # Agreement rates
        alice_key = alice_bits[key_mask]
        bob_key = bob_bits[key_mask]

        if n_matching > 0:
            agree_matching = np.mean(alice_key[matching] == bob_key[matching])
        else:
            agree_matching = None

        if n_non_matching > 0:
            agree_non_matching = np.mean(alice_key[~matching] == bob_key[~matching])
        else:
            agree_non_matching = None

        results['key_analysis'] = {
            'matching_bases': int(n_matching),
            'non_matching_bases': int(n_non_matching),
            'sifting_rate': n_matching / n_key if n_key > 0 else 0,
            'agreement_matching': float(agree_matching) if agree_matching is not None else None,
            'agreement_non_matching': float(agree_non_matching) if agree_non_matching is not None else None
        }

    # Analyze security samples (CHSH)
    if n_security > 0:
        correlators = defaultdict(list)
        for i in range(len(data['modes'])):
            if data['modes'][i] == 'security':
                setting = (int(data['alice_settings'][i]), int(data['bob_settings'][i]))
                same = 1 if alice_bits[i] == bob_bits[i] else -1
                correlators[setting].append(same)

        E = {}
        for setting in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            if setting in correlators and len(correlators[setting]) > 0:
                E[str(setting)] = {
                    'value': float(np.mean(correlators[setting])),
                    'std': float(np.std(correlators[setting]) / np.sqrt(len(correlators[setting]))),
                    'n_samples': len(correlators[setting])
                }
            else:
                E[str(setting)] = {'value': 0, 'std': 0, 'n_samples': 0}

        # CHSH value
        S = (E['(0, 0)']['value'] - E['(0, 1)']['value'] +
             E['(1, 0)']['value'] + E['(1, 1)']['value'])

        # Standard error propagation
        S_std = np.sqrt(sum(e['std']**2 for e in E.values()))

        results['security_analysis'] = {
            'correlators': E,
            'chsh_value': float(S),
            'chsh_std': float(S_std),
            'tsirelson_bound': 2.828,
            'classical_bound': 2.0,
            'bell_violation': S > 2.0,
            'efficiency': S / 2.828 if S > 0 else 0
        }

    return results


def run_ideal_simulation(n_pairs: int = 10000, key_fraction: float = 0.9,
                          visibility: float = 1.0, seed: int = 42) -> dict:
    """
    Run ideal Bell state simulation.

    Args:
        n_pairs: Number of Bell pairs
        key_fraction: Fraction for key generation
        visibility: State visibility (1.0 = perfect)
        seed: Random seed

    Returns:
        Analysis results
    """
    print(f"\n--- Ideal Simulation (visibility={visibility}) ---")

    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)

    results = analyze_qkd_data(data)
    results['simulation_params'] = {
        'n_pairs': n_pairs,
        'key_fraction': key_fraction,
        'visibility': visibility,
        'seed': seed,
        'simulator': 'IdealBellSimulator'
    }

    return results


def run_qiskit_simulation(n_pairs: int = 1000, key_fraction: float = 0.9,
                           noise_level: float = 0.0, seed: int = 42) -> dict:
    """
    Run Qiskit-based simulation with optional noise.

    Args:
        n_pairs: Number of Bell pairs
        key_fraction: Fraction for key generation
        noise_level: Depolarizing noise level (0 = ideal)
        seed: Random seed

    Returns:
        Analysis results
    """
    if not HAS_QISKIT:
        return {'error': 'Qiskit not available'}

    print(f"\n--- Qiskit Simulation (noise={noise_level}) ---")

    noise_model = None
    if noise_level > 0:
        noise_model = create_noise_model(noise_level)

    sim = QKDSimulator(noise_model=noise_model, seed=seed)
    data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)

    results = analyze_qkd_data(data)
    results['simulation_params'] = {
        'n_pairs': n_pairs,
        'key_fraction': key_fraction,
        'noise_level': noise_level,
        'seed': seed,
        'simulator': 'QKDSimulator (Qiskit Aer)'
    }

    return results


def validate_results(results: dict) -> dict:
    """
    Validate simulation results against expected values.

    Args:
        results: Analysis results

    Returns:
        Validation report
    """
    validation = {'passed': True, 'checks': []}

    # Check 1: Key fraction
    expected_key = results['simulation_params'].get('key_fraction', 0.9)
    actual_key = results['key_fraction']
    tolerance = 0.05  # 5% tolerance

    check1 = abs(actual_key - expected_key) < tolerance
    validation['checks'].append({
        'name': 'key_fraction',
        'expected': expected_key,
        'actual': actual_key,
        'tolerance': tolerance,
        'passed': check1
    })
    validation['passed'] = validation['passed'] and check1

    # Check 2: Matching basis agreement
    if 'key_analysis' in results:
        ka = results['key_analysis']
        if ka['agreement_matching'] is not None:
            check2 = ka['agreement_matching'] > 0.95  # Should be ~1.0
            validation['checks'].append({
                'name': 'matching_basis_agreement',
                'expected': '>0.95',
                'actual': ka['agreement_matching'],
                'passed': check2
            })
            validation['passed'] = validation['passed'] and check2

    # Check 3: Non-matching basis agreement
    if 'key_analysis' in results:
        ka = results['key_analysis']
        if ka['agreement_non_matching'] is not None:
            # Should be ~0.5
            check3 = 0.4 < ka['agreement_non_matching'] < 0.6
            validation['checks'].append({
                'name': 'non_matching_basis_agreement',
                'expected': '0.4-0.6',
                'actual': ka['agreement_non_matching'],
                'passed': check3
            })
            validation['passed'] = validation['passed'] and check3

    # Check 4: CHSH value
    if 'security_analysis' in results:
        sa = results['security_analysis']
        vis = results['simulation_params'].get('visibility', 1.0)
        expected_chsh = 2.828 * vis  # Approximate

        # Allow 15% deviation for statistical fluctuations
        check4 = abs(sa['chsh_value'] - expected_chsh) / expected_chsh < 0.15
        validation['checks'].append({
            'name': 'chsh_value',
            'expected': f'~{expected_chsh:.3f}',
            'actual': sa['chsh_value'],
            'passed': check4
        })
        validation['passed'] = validation['passed'] and check4

        # Check 5: Bell violation
        if vis > 0.7:  # Only expect violation for high visibility
            check5 = sa['bell_violation']
            validation['checks'].append({
                'name': 'bell_violation',
                'expected': True,
                'actual': sa['bell_violation'],
                'passed': check5
            })
            validation['passed'] = validation['passed'] and check5

    return validation


def print_results(results: dict, validation: dict):
    """Print formatted results."""
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nTotal pairs: {results['total_pairs']}")
    print(f"Key samples: {results['key_samples']} ({100*results['key_fraction']:.1f}%)")
    print(f"Security samples: {results['security_samples']}")

    if 'key_analysis' in results:
        ka = results['key_analysis']
        print(f"\nKey Analysis:")
        print(f"  Matching bases: {ka['matching_bases']}")
        print(f"  Sifting rate: {100*ka['sifting_rate']:.1f}%")
        if ka['agreement_matching'] is not None:
            print(f"  Agreement (matching): {100*ka['agreement_matching']:.2f}%")
        if ka['agreement_non_matching'] is not None:
            print(f"  Agreement (non-matching): {100*ka['agreement_non_matching']:.2f}%")

    if 'security_analysis' in results:
        sa = results['security_analysis']
        print(f"\nSecurity Analysis:")
        print(f"  CHSH S = {sa['chsh_value']:.3f} +/- {sa['chsh_std']:.3f}")
        print(f"  Classical bound: {sa['classical_bound']}")
        print(f"  Tsirelson bound: {sa['tsirelson_bound']:.3f}")
        print(f"  Bell violation: {'YES' if sa['bell_violation'] else 'NO'}")
        print(f"  Efficiency: {100*sa['efficiency']:.1f}%")

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    for check in validation['checks']:
        status = 'PASS' if check['passed'] else 'FAIL'
        print(f"  [{status}] {check['name']}: expected {check['expected']}, got {check.get('actual', 'N/A')}")

    print(f"\nOverall: {'ALL CHECKS PASSED' if validation['passed'] else 'SOME CHECKS FAILED'}")


def run_experiment():
    """Run complete Bell pair + basis experiment."""
    print("="*60)
    print("EXPERIMENT 1.1: BELL PAIR + RANDOM BASIS")
    print("="*60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Qiskit available: {HAS_QISKIT}")

    all_results = {
        'experiment': 'exp_1_1_bell_basis',
        'timestamp': datetime.now().isoformat(),
        'qiskit_available': HAS_QISKIT,
        'simulations': []
    }

    overall_passed = True

    # Test 1: Ideal simulation with high visibility
    print("\n" + "="*60)
    print("TEST 1: Ideal Simulation (visibility=1.0)")
    print("="*60)

    results1 = run_ideal_simulation(n_pairs=10000, visibility=1.0, seed=42)
    validation1 = validate_results(results1)
    print_results(results1, validation1)

    all_results['simulations'].append({
        'name': 'ideal_v1.0',
        'results': results1,
        'validation': validation1
    })
    overall_passed = overall_passed and validation1['passed']

    # Test 2: Noisy simulation (visibility=0.9)
    print("\n" + "="*60)
    print("TEST 2: Noisy Simulation (visibility=0.9)")
    print("="*60)

    results2 = run_ideal_simulation(n_pairs=10000, visibility=0.9, seed=43)
    validation2 = validate_results(results2)
    print_results(results2, validation2)

    all_results['simulations'].append({
        'name': 'noisy_v0.9',
        'results': results2,
        'validation': validation2
    })
    overall_passed = overall_passed and validation2['passed']

    # Test 3: Qiskit simulation if available
    if HAS_QISKIT:
        print("\n" + "="*60)
        print("TEST 3: Qiskit Aer Simulation (ideal)")
        print("="*60)

        results3 = run_qiskit_simulation(n_pairs=1000, noise_level=0.0, seed=44)
        if 'error' not in results3:
            validation3 = validate_results(results3)
            print_results(results3, validation3)

            all_results['simulations'].append({
                'name': 'qiskit_ideal',
                'results': results3,
                'validation': validation3
            })
            overall_passed = overall_passed and validation3['passed']

    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    n_sims = len(all_results['simulations'])
    n_passed = sum(1 for s in all_results['simulations'] if s['validation']['passed'])

    print(f"Simulations run: {n_sims}")
    print(f"Simulations passed: {n_passed}/{n_sims}")
    print(f"Overall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'simulations_run': n_sims,
        'simulations_passed': n_passed,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_1_bell_basis.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
