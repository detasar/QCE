#!/usr/bin/env python3
"""
Experiment 3.3: Sequential Tests (CUSUM & SPRT)

Sequential tests make decisions as data arrives, minimizing samples needed
for detection. Critical for real-time QKD monitoring.

Implements:
1. CUSUM (Cumulative Sum) - Page (1954)
2. SPRT (Sequential Probability Ratio Test) - Wald (1945)
3. Page-Hinkley Test - for gradual change detection

Key Metrics:
- ARL₀ (Average Run Length under H₀): Higher = fewer false alarms
- Detection delay: Samples needed after attack starts
- Power: Probability of detecting true attacks

Research Questions:
1. How quickly can we detect attacks in real-time?
2. What's the tradeoff between ARL₀ and detection delay?
3. How do sequential tests compare to batch tests (TARA-K, TARA-W, MMD)?

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.stats import beta
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import PHASE3_RESULTS
from utils.circuits import IdealBellSimulator
from utils.sifting import sift_keys
from utils.tara_integration import TARAk, qkd_to_tara_format, create_eve_data

# Ensure phase3 results directory exists
PHASE3_RESULTS.mkdir(parents=True, exist_ok=True)


class DecisionState(Enum):
    CONTINUE = "continue"
    ACCEPT_H0 = "accept_h0"  # No attack
    REJECT_H0 = "reject_h0"  # Attack detected


@dataclass
class SequentialTestResult:
    """Result of sequential test at current step."""
    step: int
    statistic: float
    decision: DecisionState
    p_value_estimate: Optional[float] = None


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) detector for QKD anomaly detection.

    Reference: Page (1954), "Continuous Inspection Schemes"

    Monitors for shift in mean of p-values.
    Under H_0 (no attack): p-values ~ U[0,1], mean = 0.5
    Under H_1 (attack): p-values shift toward 0 or 1
    """

    def __init__(self, target_mean: float = 0.5,
                 allowance: float = 0.05,
                 threshold: float = 5.0,
                 two_sided: bool = True):
        """
        Args:
            target_mean: Expected mean under H_0 (0.5 for uniform p-values)
            allowance: Slack parameter k (typically 0.5 × expected shift)
            threshold: Decision threshold h (higher = fewer false alarms)
            two_sided: Monitor both directions
        """
        self.mu_0 = target_mean
        self.k = allowance
        self.h = threshold
        self.two_sided = two_sided
        self.reset()

    def reset(self):
        """Reset detector state."""
        self.S_plus = 0.0
        self.S_minus = 0.0
        self.step = 0
        self.history = []

    def update(self, x: float) -> SequentialTestResult:
        """
        Process one observation.

        Args:
            x: New observation (p-value)

        Returns:
            SequentialTestResult with current status
        """
        self.step += 1

        # Update CUSUM statistics
        self.S_plus = max(0, self.S_plus + (x - self.mu_0) - self.k)

        if self.two_sided:
            self.S_minus = max(0, self.S_minus - (x - self.mu_0) - self.k)
            statistic = max(self.S_plus, self.S_minus)
        else:
            statistic = self.S_plus

        self.history.append(statistic)

        # Decision
        if statistic > self.h:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=statistic,
            decision=decision
        )

    def process_stream(self, p_values: np.ndarray) -> Tuple[Optional[int], List[float]]:
        """
        Process a stream of p-values.

        Returns:
            (detection_step, history) - detection_step is None if no detection
        """
        self.reset()
        detection_step = None

        for pv in p_values:
            result = self.update(pv)
            if result.decision == DecisionState.REJECT_H0 and detection_step is None:
                detection_step = result.step

        return detection_step, self.history


class SPRTDetector:
    """
    Sequential Probability Ratio Test for QKD.

    Reference: Wald (1945), "Sequential Tests of Statistical Hypotheses"

    Tests H_0: p-values ~ U[0,1] vs H_1: p-values ~ Beta(a,b)
    """

    def __init__(self,
                 alpha: float = 0.05,
                 beta_error: float = 0.05,
                 h1_params: Tuple[float, float] = (2.0, 1.0)):
        """
        Args:
            alpha: Type I error rate (false positive)
            beta_error: Type II error rate (false negative)
            h1_params: Alternative distribution params (Beta shape params)
        """
        self.alpha = alpha
        self.beta_error = beta_error
        self.h1_a, self.h1_b = h1_params

        # Wald boundaries
        self.A = np.log(beta_error / (1 - alpha))
        self.B = np.log((1 - beta_error) / alpha)

        self.reset()

    def reset(self):
        """Reset detector state."""
        self.log_likelihood_ratio = 0.0
        self.step = 0
        self.history = []

    def _log_likelihood_ratio(self, x: float) -> float:
        """
        Compute log-likelihood ratio for single observation.

        log(f_1(x) / f_0(x)) where:
        f_0(x) = 1 (uniform)
        f_1(x) = Beta(a,b) pdf
        """
        # Avoid log(0)
        x = np.clip(x, 1e-10, 1 - 1e-10)

        f1 = beta.pdf(x, self.h1_a, self.h1_b)
        f0 = 1.0  # Uniform density

        return np.log(f1 / f0)

    def update(self, x: float) -> SequentialTestResult:
        """
        Process one observation.
        """
        self.step += 1

        llr = self._log_likelihood_ratio(x)
        self.log_likelihood_ratio += llr
        self.history.append(self.log_likelihood_ratio)

        # Decision
        if self.log_likelihood_ratio <= self.A:
            decision = DecisionState.ACCEPT_H0
        elif self.log_likelihood_ratio >= self.B:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=self.log_likelihood_ratio,
            decision=decision
        )

    def process_stream(self, p_values: np.ndarray) -> Tuple[Optional[int], DecisionState, List[float]]:
        """
        Process a stream of p-values.

        Returns:
            (detection_step, final_decision, history)
        """
        self.reset()
        detection_step = None
        final_decision = DecisionState.CONTINUE

        for pv in p_values:
            result = self.update(pv)
            if result.decision != DecisionState.CONTINUE:
                detection_step = result.step
                final_decision = result.decision
                break

        return detection_step, final_decision, self.history


class PageHinkleyDetector:
    """
    Page-Hinkley test for change detection.

    Similar to CUSUM but with different update rule.
    Better for detecting gradual changes.
    """

    def __init__(self, delta: float = 0.01,
                 threshold: float = 10.0,
                 alpha: float = 0.99):
        """
        Args:
            delta: Minimum detectable change
            threshold: Detection threshold
            alpha: Forgetting factor (0.99 = slow adaptation)
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.reset()

    def reset(self):
        """Reset detector state."""
        self.sum = 0.0
        self.mean = 0.0
        self.step = 0
        self.min_sum = 0.0
        self.history = []

    def update(self, x: float) -> SequentialTestResult:
        self.step += 1

        # Update running mean
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x

        # Update sum
        self.sum += x - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)

        statistic = self.sum - self.min_sum
        self.history.append(statistic)

        if statistic > self.threshold:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=statistic,
            decision=decision
        )

    def process_stream(self, p_values: np.ndarray) -> Tuple[Optional[int], List[float]]:
        """Process a stream of p-values."""
        self.reset()
        detection_step = None

        for pv in p_values:
            result = self.update(pv)
            if result.decision == DecisionState.REJECT_H0 and detection_step is None:
                detection_step = result.step

        return detection_step, self.history


def generate_p_values(n_samples: int, attack_start: Optional[int] = None,
                      attack_strength: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Generate p-values with optional attack injection.

    Under H_0: p-values ~ U[0,1]
    Under H_1: p-values shift (simulated by Beta distribution)
    """
    np.random.seed(seed)

    if attack_start is None:
        # Pure honest data
        return np.random.uniform(0, 1, n_samples)

    # Honest portion
    honest_samples = np.random.uniform(0, 1, attack_start)

    # Attack portion - p-values shift toward 1 (anomalous)
    # Beta(a, 1) with a > 1 shifts toward 1
    attack_a = 1 + attack_strength * 3  # Scale attack strength
    attack_samples = beta.rvs(attack_a, 1, size=n_samples - attack_start)

    return np.concatenate([honest_samples, attack_samples])


def run_arl0_test(n_trials: int = 50, n_samples: int = 2000, seed: int = 42) -> dict:
    """
    Test 1: Average Run Length under H₀ (ARL₀).

    Measures how many samples until false alarm under honest conditions.
    Higher ARL₀ = fewer false positives.
    """
    print("\n" + "=" * 60)
    print("TEST 1: AVERAGE RUN LENGTH UNDER H₀ (ARL₀)")
    print("=" * 60)

    # Thresholds tuned based on sensitivity analysis
    # Higher thresholds = fewer false positives but slower detection
    detectors = {
        'CUSUM': lambda: CUSUMDetector(threshold=7.0, allowance=0.05),
        'SPRT': lambda: SPRTDetector(alpha=0.01, beta_error=0.05, h1_params=(2.0, 1.0)),
        'PageHinkley': lambda: PageHinkleyDetector(threshold=50.0, delta=0.005)
    }

    results = {}

    for name, detector_fn in detectors.items():
        run_lengths = []
        n_false_alarms = 0

        for i in range(n_trials):
            # Generate honest p-values
            p_values = generate_p_values(n_samples, attack_start=None,
                                          seed=seed + i * 100)

            detector = detector_fn()
            if name == 'SPRT':
                det_step, decision, _ = detector.process_stream(p_values)
                if decision == DecisionState.REJECT_H0:
                    n_false_alarms += 1
                    run_lengths.append(det_step)
            else:
                det_step, _ = detector.process_stream(p_values)
                if det_step is not None:
                    n_false_alarms += 1
                    run_lengths.append(det_step)

        fpr = n_false_alarms / n_trials
        avg_rl = np.mean(run_lengths) if run_lengths else n_samples

        print(f"\n{name}:")
        print(f"  False alarm rate: {100*fpr:.1f}% ({n_false_alarms}/{n_trials})")
        print(f"  ARL₀ estimate: {avg_rl:.0f} (censored at {n_samples})")

        results[name] = {
            'false_alarm_rate': fpr,
            'arl0_estimate': avg_rl,
            'n_false_alarms': n_false_alarms,
            'n_trials': n_trials,
            'acceptable': fpr < 0.10  # < 10% FPR
        }

    return results


def run_detection_delay_test(seed: int = 42) -> dict:
    """
    Test 2: Detection delay after attack starts.

    Measures how quickly each detector responds to an attack.
    """
    print("\n" + "=" * 60)
    print("TEST 2: DETECTION DELAY")
    print("=" * 60)

    attack_starts = [200, 500, 800]
    attack_strengths = [0.2, 0.4, 0.6]
    n_samples = 1200

    detectors = {
        'CUSUM': CUSUMDetector(threshold=7.0, allowance=0.05),
        'SPRT': SPRTDetector(alpha=0.01, beta_error=0.05, h1_params=(2.0, 1.0)),
        'PageHinkley': PageHinkleyDetector(threshold=50.0, delta=0.005)
    }

    results = []

    print(f"\n{'Attack':<20} {'CUSUM':<15} {'SPRT':<15} {'PageHinkley':<15}")
    print("-" * 65)

    for attack_start in attack_starts:
        for strength in attack_strengths:
            p_values = generate_p_values(n_samples, attack_start=attack_start,
                                          attack_strength=strength, seed=seed)

            row = {'attack_start': attack_start, 'attack_strength': strength}

            delays = []
            for name, detector in detectors.items():
                detector.reset()
                if name == 'SPRT':
                    det_step, _, _ = detector.process_stream(p_values)
                else:
                    det_step, _ = detector.process_stream(p_values)

                if det_step is not None:
                    delay = max(0, det_step - attack_start)
                    row[f'{name}_delay'] = delay
                    row[f'{name}_detected'] = True
                    delays.append(f"{delay:>5}")
                else:
                    row[f'{name}_delay'] = None
                    row[f'{name}_detected'] = False
                    delays.append("  N/A")

            scenario = f"Start={attack_start}, S={strength}"
            print(f"{scenario:<20} {delays[0]:<15} {delays[1]:<15} {delays[2]:<15}")

            results.append(row)

    # Summary statistics
    print("\n--- Summary ---")
    for name in detectors.keys():
        detected = [r for r in results if r.get(f'{name}_detected')]
        delays = [r[f'{name}_delay'] for r in detected]

        if delays:
            print(f"{name}: Avg delay = {np.mean(delays):.1f}, "
                  f"Detection rate = {len(detected)}/{len(results)}")
        else:
            print(f"{name}: No detections")

    return {'results': results}


def run_attack_detection_comparison(seed: int = 42) -> dict:
    """
    Test 3: Compare sequential tests with batch test (TARA-K).

    Uses actual QKD data to compare detection capabilities.
    """
    print("\n" + "=" * 60)
    print("TEST 3: COMPARISON WITH BATCH TESTS")
    print("=" * 60)

    # Generate calibration data
    sim = IdealBellSimulator(visibility=1.0, seed=seed)
    cal_qkd = sim.generate_qkd_data(n_pairs=6000, key_fraction=0.5)
    cal_sift = sift_keys(cal_qkd)
    cal_data = qkd_to_tara_format(cal_sift.security_data)

    # Create TARA-K for comparison
    tara_k = TARAk(cal_data)

    # Test scenarios with actual QKD attacks
    scenarios = [
        ('Honest', 'none', 0.0, 1.0),
        ('Intercept 0.2', 'intercept_resend', 0.2, 1.0),
        ('Intercept 0.4', 'intercept_resend', 0.4, 1.0),
        ('Decorrelation 0.3', 'decorrelation', 0.3, 1.0),
        ('Visibility 0.85', 'visibility', 0.0, 0.85),
        ('Visibility 0.75', 'visibility', 0.0, 0.75),
    ]

    results = []

    print(f"\n{'Scenario':<20} {'CUSUM':<10} {'SPRT':<10} {'PH':<10} {'TARA-K':<10}")
    print("-" * 60)

    for scenario_name, attack_type, noise_level, visibility in scenarios:
        # Generate test data
        test_sim = IdealBellSimulator(visibility=visibility, seed=seed + 1000)
        test_qkd = test_sim.generate_qkd_data(n_pairs=3000, key_fraction=0.5)
        test_sift = sift_keys(test_qkd)
        test_data = qkd_to_tara_format(test_sift.security_data)

        if attack_type == 'intercept_resend':
            test_data = create_eve_data(test_data, attack_type='intercept_resend',
                                         noise_level=noise_level, seed=seed + 2000)
        elif attack_type == 'decorrelation':
            test_data = create_eve_data(test_data, attack_type='decorrelation',
                                         noise_level=noise_level, seed=seed + 2000)

        # Compute p-values for sequential tests
        scores = []
        for i in range(len(test_data['x'])):
            p = tara_k.cond_probs[(test_data['x'][i], test_data['y'][i])][test_data['a'][i], test_data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        scores = np.array(scores)

        p_values = []
        for s in scores:
            rank = np.sum(tara_k.cal_scores >= s) + 1
            p_values.append(rank / (len(tara_k.cal_scores) + 1))
        p_values = np.array(p_values)

        # Test with sequential detectors
        cusum = CUSUMDetector(threshold=7.0, allowance=0.05)
        sprt = SPRTDetector(alpha=0.01, beta_error=0.05, h1_params=(2.0, 1.0))
        ph = PageHinkleyDetector(threshold=50.0, delta=0.005)

        cusum_det, _ = cusum.process_stream(p_values)
        sprt_det, sprt_decision, _ = sprt.process_stream(p_values)
        ph_det, _ = ph.process_stream(p_values)

        # Test with TARA-K
        tara_k_result = tara_k.test(test_data, threshold=0.2)

        # Results
        cusum_status = "DET" if cusum_det else "-"
        sprt_status = "DET" if sprt_decision == DecisionState.REJECT_H0 else "-"
        ph_status = "DET" if ph_det else "-"
        tarak_status = "DET" if tara_k_result.detected else "-"

        print(f"{scenario_name:<20} {cusum_status:<10} {sprt_status:<10} {ph_status:<10} {tarak_status:<10}")

        results.append({
            'scenario': scenario_name,
            'attack_type': attack_type,
            'noise_level': noise_level,
            'visibility': visibility,
            'cusum_detected': cusum_det is not None,
            'sprt_detected': sprt_decision == DecisionState.REJECT_H0,
            'ph_detected': ph_det is not None,
            'tara_k_detected': tara_k_result.detected
        })

    # Count detections
    n_attacks = sum(1 for r in results if r['attack_type'] != 'none')
    cusum_dets = sum(1 for r in results if r['cusum_detected'] and r['attack_type'] != 'none')
    sprt_dets = sum(1 for r in results if r['sprt_detected'] and r['attack_type'] != 'none')
    ph_dets = sum(1 for r in results if r['ph_detected'] and r['attack_type'] != 'none')
    tarak_dets = sum(1 for r in results if r['tara_k_detected'] and r['attack_type'] != 'none')

    print(f"\n--- Detection Summary ---")
    print(f"CUSUM: {cusum_dets}/{n_attacks}")
    print(f"SPRT: {sprt_dets}/{n_attacks}")
    print(f"Page-Hinkley: {ph_dets}/{n_attacks}")
    print(f"TARA-K: {tarak_dets}/{n_attacks}")

    return {
        'results': results,
        'cusum_detections': cusum_dets,
        'sprt_detections': sprt_dets,
        'ph_detections': ph_dets,
        'tara_k_detections': tarak_dets,
        'attack_count': n_attacks
    }


def run_parameter_sensitivity(seed: int = 42) -> dict:
    """
    Test 4: Parameter sensitivity analysis.

    Tests how different thresholds affect detection and FPR.
    """
    print("\n" + "=" * 60)
    print("TEST 4: PARAMETER SENSITIVITY")
    print("=" * 60)

    n_trials = 30
    n_samples = 1000

    # CUSUM threshold sweep
    cusum_thresholds = [2.0, 3.0, 5.0, 7.0, 10.0]
    cusum_results = []

    print("\nCUSUM Threshold Sensitivity:")
    print(f"{'Threshold':<15} {'FPR':<15} {'Avg Delay':<15}")
    print("-" * 45)

    for h in cusum_thresholds:
        n_fp = 0
        delays = []

        for i in range(n_trials):
            # Test on honest data
            p_honest = generate_p_values(n_samples, attack_start=None, seed=seed + i)
            det, _ = CUSUMDetector(threshold=h).process_stream(p_honest)
            if det is not None:
                n_fp += 1

            # Test on attack data
            p_attack = generate_p_values(n_samples, attack_start=300, attack_strength=0.4,
                                          seed=seed + 1000 + i)
            det, _ = CUSUMDetector(threshold=h).process_stream(p_attack)
            if det is not None:
                delays.append(max(0, det - 300))

        fpr = n_fp / n_trials
        avg_delay = np.mean(delays) if delays else n_samples

        print(f"{h:<15.1f} {100*fpr:<15.1f}% {avg_delay:<15.1f}")

        cusum_results.append({
            'threshold': h,
            'fpr': fpr,
            'avg_delay': avg_delay,
            'detection_rate': len(delays) / n_trials
        })

    return {'cusum_sensitivity': cusum_results}


def validate_results(all_results: dict) -> dict:
    """Validate experiment results."""
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    checks = []

    # Check 1: ARL₀ - low false positive rate under H₀
    # At least 2/3 methods should have acceptable FPR
    arl = all_results.get('arl0', {})
    fpr_acceptable_count = 0
    fpr_details = []
    for name in ['CUSUM', 'SPRT', 'PageHinkley']:
        det_result = arl.get(name, {})
        fpr = det_result.get('false_alarm_rate', 1)
        if fpr < 0.10:
            fpr_acceptable_count += 1
        fpr_details.append(f"{name}:{100*fpr:.1f}%")

    checks.append({
        'name': 'At least 2/3 methods have FPR < 10%',
        'passed': fpr_acceptable_count >= 2,
        'detail': f"{fpr_acceptable_count}/3 pass ({', '.join(fpr_details)})"
    })

    # Check 2: Detection capability
    comparison = all_results.get('comparison', {})
    n_attacks = comparison.get('attack_count', 0)

    # At least one sequential test should detect some attacks
    cusum_dets = comparison.get('cusum_detections', 0)
    sprt_dets = comparison.get('sprt_detections', 0)
    ph_dets = comparison.get('ph_detections', 0)
    best_sequential = max(cusum_dets, sprt_dets, ph_dets)

    checks.append({
        'name': 'Sequential tests detect attacks (at least 1 attack)',
        'passed': best_sequential >= 1 if n_attacks > 0 else True,
        'detail': f"Best: {best_sequential}/{n_attacks}"
    })

    # Check 3: Comparison with TARA-K
    tarak_dets = comparison.get('tara_k_detections', 0)
    checks.append({
        'name': 'Sequential tests competitive with TARA-K',
        'passed': best_sequential >= tarak_dets,
        'detail': f"Sequential best: {best_sequential}, TARA-K: {tarak_dets}"
    })

    # Print validation results
    for check in checks:
        status = "PASS" if check['passed'] else "FAIL"
        print(f"  [{status}] {check['name']}")
        print(f"        {check['detail']}")

    all_passed = all(c['passed'] for c in checks)
    n_passed = sum(1 for c in checks if c['passed'])
    print(f"\nPassed: {n_passed}/{len(checks)} checks")
    print(f"Overall: {'SUCCESS' if all_passed else 'NEEDS REVIEW'}")

    return {'checks': checks, 'all_passed': all_passed}


def run_experiment():
    """Run complete Sequential Tests experiment."""
    print("=" * 60)
    print("EXPERIMENT 3.3: SEQUENTIAL TESTS (CUSUM, SPRT, PAGE-HINKLEY)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    all_results = {
        'experiment': 'exp_3_3_sequential_tests',
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: ARL₀
    print("\n" + "#" * 60)
    print("RUNNING TEST 1: ARL₀")
    print("#" * 60)
    arl_results = run_arl0_test(n_trials=30, seed=42)
    all_results['tests']['arl0'] = arl_results

    # Test 2: Detection delay
    print("\n" + "#" * 60)
    print("RUNNING TEST 2: DETECTION DELAY")
    print("#" * 60)
    delay_results = run_detection_delay_test(seed=42)
    all_results['tests']['detection_delay'] = delay_results

    # Test 3: Comparison with batch tests
    print("\n" + "#" * 60)
    print("RUNNING TEST 3: COMPARISON WITH BATCH TESTS")
    print("#" * 60)
    comparison_results = run_attack_detection_comparison(seed=42)
    all_results['tests']['comparison'] = comparison_results

    # Test 4: Parameter sensitivity
    print("\n" + "#" * 60)
    print("RUNNING TEST 4: PARAMETER SENSITIVITY")
    print("#" * 60)
    sensitivity_results = run_parameter_sensitivity(seed=42)
    all_results['tests']['sensitivity'] = sensitivity_results

    # Validation
    validation = validate_results(all_results['tests'])
    all_results['validation'] = validation

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print("\nKey Findings:")
    for name in ['CUSUM', 'SPRT', 'PageHinkley']:
        det = arl_results.get(name, {})
        print(f"  - {name} FPR: {100*det.get('false_alarm_rate', 0):.1f}%")

    print(f"\nAttack Detection (out of {comparison_results['attack_count']}):")
    print(f"  - CUSUM: {comparison_results['cusum_detections']}")
    print(f"  - SPRT: {comparison_results['sprt_detections']}")
    print(f"  - Page-Hinkley: {comparison_results['ph_detections']}")
    print(f"  - TARA-K (baseline): {comparison_results['tara_k_detections']}")

    all_results['summary'] = {
        'cusum_fpr': arl_results.get('CUSUM', {}).get('false_alarm_rate', 0),
        'sprt_fpr': arl_results.get('SPRT', {}).get('false_alarm_rate', 0),
        'ph_fpr': arl_results.get('PageHinkley', {}).get('false_alarm_rate', 0),
        'cusum_detections': comparison_results['cusum_detections'],
        'sprt_detections': comparison_results['sprt_detections'],
        'ph_detections': comparison_results['ph_detections'],
        'tara_k_detections': comparison_results['tara_k_detections'],
        'attack_count': comparison_results['attack_count'],
        'validation_passed': validation['all_passed']
    }

    # Save results
    output_file = PHASE3_RESULTS / 'exp_3_3_sequential_tests.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
