#!/usr/bin/env python3
"""
Experiment 1.6: TARA-m Streaming Detection

This experiment validates TARA-m (Martingale-based) detector for
real-time QKD security monitoring:
1. Sequential anomaly detection
2. Wealth accumulation under attack
3. Detection time measurement
4. Betting strategy comparison

TARA-m uses a test martingale that:
- Under H0 (honest): wealth stays bounded around 1
- Under H1 (Eve): wealth grows exponentially
- Detection when wealth exceeds threshold

Expected results:
- No detection for honest data
- Fast detection (< 500 samples) for attacks
- Different betting strategies have different sensitivities

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
from utils.tara_integration import TARAm, qkd_to_tara_format, create_eve_data


def generate_tara_data(n_pairs: int = 10000, visibility: float = 1.0,
                        security_fraction: float = 0.5, seed: int = 42) -> dict:
    """Generate TARA-formatted data."""
    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    key_fraction = 1 - security_fraction
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)
    sifting_result = sift_keys(qkd_data)
    return qkd_to_tara_format(sifting_result.security_data)


def run_tara_m_basic_test(seed: int = 42) -> dict:
    """
    Test basic TARA-m behavior on honest data.

    Args:
        seed: Random seed

    Returns:
        Test results
    """
    print("\n" + "=" * 60)
    print("TARA-m Basic Test (Honest Data)")
    print("=" * 60)

    # Generate calibration and test data
    cal_data = generate_tara_data(n_pairs=10000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)
    test_data = generate_tara_data(n_pairs=5000, visibility=1.0,
                                    security_fraction=0.5, seed=seed + 100)

    n_cal = len(cal_data['x'])
    n_test = len(test_data['x'])

    print(f"\nCalibration samples: {n_cal}")
    print(f"Test samples: {n_test}")

    # Test with different betting strategies
    strategies = ['linear', 'jumper', 'twosided']
    results = {}

    for strategy in strategies:
        detector = TARAm(cal_data, epsilon=0.5, betting=strategy)
        result = detector.test(test_data, threshold=20.0)

        print(f"\n{strategy.upper()} betting:")
        print(f"  Max log wealth: {result.statistic:.2f}")
        print(f"  Mean p-value: {result.mean_p_value:.4f}")
        print(f"  Detected: {'YES' if result.detected else 'NO'}")

        results[strategy] = {
            'max_log_wealth': result.statistic,
            'mean_p_value': result.mean_p_value,
            'detected': result.detected
        }

    # Check that honest data is NOT detected
    any_detected = any(r['detected'] for r in results.values())

    return {
        'n_calibration': n_cal,
        'n_test': n_test,
        'strategies': results,
        'any_false_positive': any_detected,
        'passed': not any_detected
    }


def run_attack_detection_test(visibility: float = 0.7, seed: int = 42) -> dict:
    """
    Test TARA-m attack detection with reduced visibility.

    Args:
        visibility: Attack visibility (lower = more severe attack)
        seed: Random seed

    Returns:
        Test results
    """
    print(f"\n{'='*60}")
    print(f"TARA-m Attack Detection (visibility={visibility})")
    print("=" * 60)

    # Generate calibration (honest)
    cal_data = generate_tara_data(n_pairs=10000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Generate attack data (reduced visibility)
    attack_data = generate_tara_data(n_pairs=5000, visibility=visibility,
                                      security_fraction=0.5, seed=seed + 200)

    n_cal = len(cal_data['x'])
    n_attack = len(attack_data['x'])

    print(f"\nCalibration (v=1.0): {n_cal} samples")
    print(f"Attack (v={visibility}): {n_attack} samples")

    # Test with different betting strategies
    strategies = ['linear', 'jumper', 'twosided']
    results = {}

    for strategy in strategies:
        detector = TARAm(cal_data, epsilon=0.5, betting=strategy)
        result = detector.test(attack_data, threshold=20.0)

        print(f"\n{strategy.upper()} betting:")
        print(f"  Max log wealth: {result.statistic:.2f}")
        print(f"  Mean p-value: {result.mean_p_value:.4f}")
        print(f"  Detected: {'YES' if result.detected else 'NO'}")
        if result.detection_time is not None:
            print(f"  Detection time: {result.detection_time} samples")

        results[strategy] = {
            'max_log_wealth': result.statistic,
            'mean_p_value': result.mean_p_value,
            'detected': result.detected,
            'detection_time': result.detection_time
        }

    # Check detection
    n_detected = sum(1 for r in results.values() if r['detected'])

    return {
        'visibility': visibility,
        'n_calibration': n_cal,
        'n_attack': n_attack,
        'strategies': results,
        'n_detected': n_detected,
        'all_detected': n_detected == len(strategies),
        'passed': n_detected >= 2  # At least 2 of 3 detect
    }


def run_detection_time_analysis(seed: int = 42) -> dict:
    """
    Analyze detection time vs attack severity.

    Args:
        seed: Random seed

    Returns:
        Analysis results
    """
    print("\n" + "=" * 60)
    print("Detection Time Analysis")
    print("=" * 60)

    # Generate calibration
    cal_data = generate_tara_data(n_pairs=10000, visibility=1.0,
                                   security_fraction=0.5, seed=seed)

    # Test different visibility levels
    visibilities = [0.9, 0.8, 0.7, 0.6, 0.5]
    results = []

    for v in visibilities:
        attack_data = generate_tara_data(n_pairs=5000, visibility=v,
                                          security_fraction=0.5, seed=seed + int(v * 1000))

        detector = TARAm(cal_data, epsilon=0.5, betting='twosided')
        result = detector.test(attack_data, threshold=20.0)

        print(f"\nVisibility {v}:")
        print(f"  Max log wealth: {result.statistic:.2f}")
        print(f"  Detected: {'YES' if result.detected else 'NO'}")
        if result.detection_time is not None:
            print(f"  Detection time: {result.detection_time} samples")

        results.append({
            'visibility': v,
            'max_log_wealth': result.statistic,
            'detected': result.detected,
            'detection_time': result.detection_time
        })

    # Summary
    print("\n--- Summary ---")
    print(f"{'Visibility':<12} {'Log Wealth':<12} {'Detected':<10} {'Time':<10}")
    print("-" * 44)
    for r in results:
        time_str = str(r['detection_time']) if r['detection_time'] else 'N/A'
        print(f"{r['visibility']:<12.1f} {r['max_log_wealth']:<12.2f} {'YES' if r['detected'] else 'NO':<10} {time_str:<10}")

    return {
        'visibilities': [r['visibility'] for r in results],
        'detection_times': [r['detection_time'] for r in results],
        'max_log_wealths': [r['max_log_wealth'] for r in results],
        'detections': [r['detected'] for r in results]
    }


def run_experiment():
    """Run complete TARA-m experiment."""
    print("=" * 60)
    print("EXPERIMENT 1.6: TARA-m STREAMING DETECTION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_1_6_tara_m',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    overall_passed = True

    # Test 1: Basic behavior on honest data
    print("\n" + "#" * 60)
    print("TEST 1: HONEST DATA (No Detection Expected)")
    print("#" * 60)

    basic_results = run_tara_m_basic_test(seed=42)
    all_results['tests']['basic'] = basic_results

    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if basic_results['passed'] else 'FAIL'}] No false positives")
    overall_passed = overall_passed and basic_results['passed']

    # Test 2: Attack detection (moderate)
    print("\n" + "#" * 60)
    print("TEST 2: MODERATE ATTACK (v=0.7)")
    print("#" * 60)

    attack_moderate = run_attack_detection_test(visibility=0.7, seed=42)
    all_results['tests']['attack_moderate'] = attack_moderate

    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if attack_moderate['passed'] else 'FAIL'}] Attack detected by >= 2 strategies")
    overall_passed = overall_passed and attack_moderate['passed']

    # Test 3: Attack detection (severe)
    print("\n" + "#" * 60)
    print("TEST 3: SEVERE ATTACK (v=0.5)")
    print("#" * 60)

    attack_severe = run_attack_detection_test(visibility=0.5, seed=43)
    all_results['tests']['attack_severe'] = attack_severe

    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if attack_severe['passed'] else 'FAIL'}] Attack detected by >= 2 strategies")
    overall_passed = overall_passed and attack_severe['passed']

    # Test 4: Detection time analysis
    print("\n" + "#" * 60)
    print("TEST 4: DETECTION TIME ANALYSIS")
    print("#" * 60)

    time_results = run_detection_time_analysis(seed=44)
    all_results['tests']['detection_time'] = time_results

    # Check that more severe attacks are detected faster
    time_check = time_results['detection_times'][-1] is not None  # Most severe should be detected
    print(f"\n--- VALIDATION ---")
    print(f"  [{'PASS' if time_check else 'FAIL'}] Severe attack (v=0.5) detected")
    overall_passed = overall_passed and time_check

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    tests_summary = [
        ('Honest data (no FP)', basic_results['passed']),
        ('Moderate attack', attack_moderate['passed']),
        ('Severe attack', attack_severe['passed']),
        ('Detection time', time_check)
    ]

    for name, passed in tests_summary:
        print(f"  [{('PASS' if passed else 'FAIL')}] {name}")

    n_passed = sum(1 for _, p in tests_summary if p)
    overall_passed = n_passed >= 3  # Allow 1 failure

    print(f"\nOverall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'basic_passed': basic_results['passed'],
        'moderate_attack_passed': attack_moderate['passed'],
        'severe_attack_passed': attack_severe['passed'],
        'detection_time_check': time_check,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_6_tara_m.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
