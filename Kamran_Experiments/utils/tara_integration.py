"""
TARA Integration for QKD Security Monitoring

This module provides integration with TARA (Testing Anomalous Relative to
Reference) conformal prediction framework for real-time anomaly detection
in QKD systems.

TARA-k: KS-test based batch detector
- Tests if p-value distribution deviates from Uniform(0,1)
- Good for offline analysis of complete sessions

TARA-m: Martingale-based sequential detector
- Builds wealth process from betting on p-values
- Good for real-time streaming detection

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal


@dataclass
class TARAResult:
    """Result from TARA detector."""
    detected: bool           # Was anomaly detected?
    statistic: float         # KS statistic or max log wealth
    threshold: float         # Detection threshold used
    p_values: np.ndarray     # Conformal p-values
    mean_p_value: float      # Average p-value
    detection_time: Optional[int]  # When detection occurred (TARAm)

    def to_dict(self) -> dict:
        return {
            'detected': self.detected,
            'statistic': self.statistic,
            'threshold': self.threshold,
            'mean_p_value': self.mean_p_value,
            'detection_time': self.detection_time,
            'n_samples': len(self.p_values)
        }


def qkd_to_tara_format(security_data: dict) -> dict:
    """
    Convert QKD security data to TARA format.

    TARA expects:
        x: Alice's measurement setting (0 or 1)
        y: Bob's measurement setting (0 or 1)
        a: Alice's outcome (0 or 1)
        b: Bob's outcome (0 or 1)

    Args:
        security_data: Dictionary from sifting with security samples

    Returns:
        Dictionary in TARA format
    """
    return {
        'x': np.array(security_data['alice_settings']),
        'y': np.array(security_data['bob_settings']),
        'a': np.array(security_data['alice_outcomes']),
        'b': np.array(security_data['bob_outcomes'])
    }


class TARAk:
    """
    TARA-k: KS test based detector.

    Tests if the p-value distribution deviates from Uniform(0,1).
    Under H0 (exchangeable quantum data), p-values should be uniform.
    Under H1 (eavesdropper), p-values deviate.
    """

    def __init__(self, calibration_data: dict):
        """
        Initialize with calibration data.

        Args:
            calibration_data: Dict with 'x', 'y', 'a', 'b' arrays from honest quantum source
        """
        self.cond_probs = self._learn_probs(calibration_data)
        self.cal_scores = self._compute_scores(calibration_data)

    def _learn_probs(self, data: dict) -> dict:
        """Learn P(a,b|x,y) from calibration data with Laplace smoothing."""
        x, y, a, b = data['x'], data['y'], data['a'], data['b']
        probs = {}

        for xi in [0, 1]:
            for yi in [0, 1]:
                mask = (x == xi) & (y == yi)
                if mask.sum() > 0:
                    counts = np.zeros((2, 2))
                    for ai in [0, 1]:
                        for bi in [0, 1]:
                            counts[ai, bi] = np.sum((a[mask] == ai) & (b[mask] == bi))
                    counts += 1  # Laplace smoothing
                    probs[(xi, yi)] = counts / counts.sum()
                else:
                    probs[(xi, yi)] = np.ones((2, 2)) / 4

        return probs

    def _compute_scores(self, data: dict) -> np.ndarray:
        """Compute nonconformity scores: -log P(a,b|x,y)."""
        scores = []
        for i in range(len(data['x'])):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        return np.array(scores)

    def compute_p_values(self, data: dict) -> np.ndarray:
        """
        Compute conformal p-values for each sample.

        p-value = (# calibration scores >= test score + 1) / (n_cal + 1)
        """
        scores = self._compute_scores(data)
        p_values = []

        for s in scores:
            rank = np.sum(self.cal_scores >= s) + 1
            p_values.append(rank / (len(self.cal_scores) + 1))

        return np.array(p_values)

    def test(self, data: dict, threshold: float = 0.2) -> TARAResult:
        """
        Test data for anomalies using KS test.

        Args:
            data: Test data in TARA format
            threshold: KS statistic threshold for detection

        Returns:
            TARAResult with detection info
        """
        p_values = self.compute_p_values(data)
        ks_stat, ks_pval = stats.kstest(p_values, 'uniform')

        return TARAResult(
            detected=ks_stat > threshold,
            statistic=ks_stat,
            threshold=threshold,
            p_values=p_values,
            mean_p_value=np.mean(p_values),
            detection_time=None  # Not applicable for batch test
        )


class TARAm:
    """
    TARA-m: Martingale-based sequential detector.

    Uses conformal p-values to construct a test martingale.
    Under H0, wealth stays bounded. Under H1 (attack), wealth grows.
    """

    def __init__(self,
                 calibration_data: dict,
                 epsilon: float = 0.5,
                 betting: Literal['linear', 'jumper', 'twosided'] = 'linear'):
        """
        Initialize with calibration data.

        Args:
            calibration_data: Dict with 'x', 'y', 'a', 'b' arrays
            epsilon: Betting intensity (0 < epsilon < 1)
            betting: Betting strategy
        """
        self.epsilon = epsilon
        self.betting = betting
        self.cond_probs = self._learn_probs(calibration_data)
        self.cal_scores = self._compute_scores(calibration_data)

    def _learn_probs(self, data: dict) -> dict:
        """Learn P(a,b|x,y) from calibration data."""
        x, y, a, b = data['x'], data['y'], data['a'], data['b']
        probs = {}

        for xi in [0, 1]:
            for yi in [0, 1]:
                mask = (x == xi) & (y == yi)
                if mask.sum() > 0:
                    counts = np.zeros((2, 2))
                    for ai in [0, 1]:
                        for bi in [0, 1]:
                            counts[ai, bi] = np.sum((a[mask] == ai) & (b[mask] == bi))
                    counts += 1  # Laplace smoothing
                    probs[(xi, yi)] = counts / counts.sum()
                else:
                    probs[(xi, yi)] = np.ones((2, 2)) / 4

        return probs

    def _compute_scores(self, data: dict) -> np.ndarray:
        """Compute nonconformity scores."""
        scores = []
        for i in range(len(data['x'])):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        return np.array(scores)

    def _compute_bet(self, pv: float) -> float:
        """Compute betting multiplier based on p-value."""
        pv = max(pv, 0.001)
        pv = min(pv, 0.999)

        if self.betting == 'linear':
            return 1.0 + self.epsilon * (1.0 - 2.0 * pv)
        elif self.betting == 'jumper':
            return (1.0 - self.epsilon) * (pv ** (-self.epsilon))
        elif self.betting == 'twosided':
            deviation = abs(0.5 - pv) * 2
            return 1.0 + self.epsilon * deviation
        else:
            return 1.0

    def test(self, data: dict, threshold: float = 20.0) -> TARAResult:
        """
        Test data using martingale.

        Args:
            data: Test data in TARA format
            threshold: Wealth threshold for detection

        Returns:
            TARAResult with detection info
        """
        n = len(data['x'])
        wealth = 1.0
        max_wealth = 1.0
        wealth_history = [1.0]
        p_values = []

        for i in range(n):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            score = -np.log(p + 1e-10)

            # Conformal p-value
            rank = np.sum(self.cal_scores >= score) + 1
            pv = rank / (len(self.cal_scores) + 1)
            p_values.append(pv)

            # Betting
            bet = self._compute_bet(pv)
            bet = max(bet, 0.01)

            wealth *= bet
            max_wealth = max(max_wealth, wealth)
            wealth_history.append(min(wealth, 1e100))

        # Detection time
        warr = np.array(wealth_history)
        det_idx = np.where(warr >= threshold)[0]
        det_time = int(det_idx[0]) if len(det_idx) > 0 else None

        log_wealth = np.log10(max_wealth + 1e-10)
        log_wealth = min(log_wealth, 100)

        return TARAResult(
            detected=max_wealth >= threshold,
            statistic=log_wealth,
            threshold=np.log10(threshold),
            p_values=np.array(p_values),
            mean_p_value=np.mean(p_values),
            detection_time=det_time
        )


def create_eve_data(honest_data: dict, attack_type: str = 'intercept_resend',
                    noise_level: float = 0.1, seed: int = None) -> dict:
    """
    Create simulated eavesdropper attack data.

    Args:
        honest_data: Original honest data
        attack_type: Type of attack ('intercept_resend', 'pns', 'decorrelation')
        noise_level: Attack strength
        seed: Random seed

    Returns:
        Modified data simulating attack
    """
    rng = np.random.RandomState(seed)
    eve_data = {k: v.copy() for k, v in honest_data.items()}

    n = len(honest_data['a'])

    if attack_type == 'intercept_resend':
        # Eve measures and resends - introduces QBER and decorrelation
        # This breaks the quantum correlation pattern
        for i in range(n):
            if rng.random() < noise_level:
                # Eve's measurement destroys entanglement
                # Result becomes random (decorrelated)
                eve_data['a'][i] = rng.randint(0, 2)
                eve_data['b'][i] = rng.randint(0, 2)

    elif attack_type == 'pns':
        # Photon number splitting - breaks correlations on subset
        mask = rng.random(n) < noise_level
        n_affected = np.sum(mask)
        # Eve learns some bits, but can't control Bob's outcome
        eve_data['b'][mask] = rng.randint(0, 2, n_affected)

    elif attack_type == 'decorrelation':
        # Pure decorrelation attack - replaces quantum data with classical
        # This is detectable because it changes P(a,b|x,y)
        for i in range(n):
            if rng.random() < noise_level:
                # Replace with uniformly random outcomes
                eve_data['a'][i] = rng.randint(0, 2)
                eve_data['b'][i] = rng.randint(0, 2)

    return eve_data


class TARAk_Correlation:
    """
    TARA-k with correlation-based scoring for QKD.

    Uses CHSH-like correlator as the nonconformity score,
    which is more sensitive to eavesdropper attacks.
    """

    def __init__(self, calibration_data: dict, window_size: int = 50):
        """
        Initialize with calibration data.

        Args:
            calibration_data: TARA-format data from honest source
            window_size: Window for computing local correlations
        """
        self.window_size = window_size
        self.cal_correlations = self._compute_window_correlations(calibration_data)

    def _compute_window_correlations(self, data: dict) -> np.ndarray:
        """Compute windowed correlation scores."""
        n = len(data['x'])
        if n < self.window_size:
            return np.array([self._compute_chsh_like(data)])

        correlations = []
        for i in range(0, n - self.window_size + 1, self.window_size // 2):
            window_data = {
                k: v[i:i + self.window_size] for k, v in data.items()
            }
            corr = self._compute_chsh_like(window_data)
            correlations.append(corr)

        return np.array(correlations)

    def _compute_chsh_like(self, data: dict) -> float:
        """
        Compute CHSH-like score from data window.

        Returns absolute deviation from expected quantum correlations.
        """
        x, y, a, b = data['x'], data['y'], data['a'], data['b']

        # Compute correlators for each setting
        correlators = {}
        for xi in [0, 1]:
            for yi in [0, 1]:
                mask = (x == xi) & (y == yi)
                if mask.sum() >= 5:
                    same = np.sum(a[mask] == b[mask])
                    E = (2 * same - mask.sum()) / mask.sum()
                    correlators[(xi, yi)] = E
                else:
                    correlators[(xi, yi)] = 0

        # CHSH-like value
        S = abs(correlators.get((0, 0), 0) - correlators.get((0, 1), 0) +
                correlators.get((1, 0), 0) + correlators.get((1, 1), 0))

        return S

    def test(self, data: dict, threshold: float = 0.15) -> TARAResult:
        """
        Test data for anomalies.

        Compares test correlations to calibration distribution.
        """
        test_correlations = self._compute_window_correlations(data)

        if len(test_correlations) == 0:
            return TARAResult(
                detected=False, statistic=0, threshold=threshold,
                p_values=np.array([0.5]), mean_p_value=0.5, detection_time=None
            )

        # Compute p-values by comparing to calibration
        p_values = []
        for tc in test_correlations:
            # P-value: fraction of calibration correlations <= test correlation
            # (Lower correlation = more anomalous)
            pv = np.mean(self.cal_correlations <= tc)
            p_values.append(pv)

        p_values = np.array(p_values)

        # KS test for uniformity
        ks_stat, _ = stats.kstest(p_values, 'uniform')

        # Also check if correlations are systematically lower
        mean_cal = np.mean(self.cal_correlations)
        mean_test = np.mean(test_correlations)
        correlation_drop = mean_cal - mean_test

        # Combined detection
        detected = ks_stat > threshold or correlation_drop > 0.3

        return TARAResult(
            detected=detected,
            statistic=ks_stat,
            threshold=threshold,
            p_values=p_values,
            mean_p_value=np.mean(p_values),
            detection_time=None
        )


def evaluate_detector_auc(detector, honest_test: dict, eve_test: dict) -> float:
    """
    Evaluate detector AUC (Area Under ROC Curve).

    Args:
        detector: TARAk or TARAm instance
        honest_test: Honest test data
        eve_test: Attack test data

    Returns:
        AUC value (0.5 = random, 1.0 = perfect)
    """
    honest_result = detector.test(honest_test)
    eve_result = detector.test(eve_test)

    # Use statistic as score
    honest_score = honest_result.statistic
    eve_score = eve_result.statistic

    # AUC for single sample comparison
    if eve_score > honest_score:
        return 1.0
    elif eve_score < honest_score:
        return 0.0
    else:
        return 0.5
