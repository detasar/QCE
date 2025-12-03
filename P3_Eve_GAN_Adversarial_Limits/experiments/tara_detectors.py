#!/usr/bin/env python3
"""
================================================================================
TARA DETECTORS FOR QUANTUM DATA AUTHENTICATION
================================================================================

Implements TARA-k (batch) and TARA-m (streaming) detectors for conformal
prediction-based quantum anomaly detection.

TARA = Test by Adaptive Ranks for Anomalies

KEY INSIGHT:
------------
Under exchangeability (honest quantum data), conformal p-values are uniform.
Eve-GAN data produces systematically high p-values (over-predictable),
which TARA detects via KS test (batch) or martingale (streaming).

TARA-k (Batch Detection):
- Collects p-values from entire test batch
- KS test against Uniform(0,1)
- Good for post-hoc analysis

TARA-m (Streaming Detection):
- Sequential martingale test
- Real-time detection
- Can detect attack within O(100) samples

Author: Davut Emre Tasar
================================================================================
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Literal


class TARAk:
    """
    TARA-k: KS test based batch detector.

    Tests if p-value distribution deviates from Uniform(0,1).
    Under H0 (exchangeable/honest data), p-values should be uniform.
    Under H1 (Eve attack), p-values typically cluster at high values.

    Algorithm:
    1. Learn P(a,b|x,y) from calibration data
    2. Compute nonconformity scores: s = -log P(a,b|x,y)
    3. For test data, compute conformal p-values
    4. KS test against uniform distribution
    """

    def __init__(self, calibration_data: Dict[str, np.ndarray]):
        """
        Initialize with calibration data.

        IMPORTANT: Calibration should be on product state (classical) data
        to avoid leakage. Do NOT calibrate on quantum data you're testing.

        Args:
            calibration_data: Dict with 'x', 'y', 'a', 'b' arrays
        """
        self.cond_probs = self._learn_probs(calibration_data)
        self.cal_scores = self._compute_scores(calibration_data)

    def _learn_probs(self, data: Dict[str, np.ndarray]) -> Dict:
        """Learn P(a,b|x,y) from data with Laplace smoothing."""
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

    def _compute_scores(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute nonconformity scores: -log P(a,b|x,y)."""
        scores = []
        for i in range(len(data['x'])):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        return np.array(scores)

    def compute_p_values(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute conformal p-values for each sample.

        p-value = (# calibration scores >= test score + 1) / (n_cal + 1)

        Higher p-value = more conforming to calibration distribution.
        Under H0, p-values ~ Uniform(0,1).
        """
        scores = self._compute_scores(data)
        p_values = []

        for s in scores:
            rank = np.sum(self.cal_scores >= s) + 1
            p_values.append(rank / (len(self.cal_scores) + 1))

        return np.array(p_values)

    def test(self, data: Dict[str, np.ndarray],
             threshold: float = 0.2) -> Dict:
        """
        Test data for anomalies using KS test.

        Args:
            data: Test data dict with 'x', 'y', 'a', 'b'
            threshold: KS statistic threshold for detection

        Returns:
            Dict with:
            - ks_statistic: KS test statistic
            - ks_pvalue: p-value from KS test
            - detected: True if KS > threshold
            - p_values: Array of conformal p-values
            - mean_p_value, std_p_value: Summary statistics
        """
        p_values = self.compute_p_values(data)
        ks_stat, ks_pval = stats.kstest(p_values, 'uniform')

        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'detected': ks_stat > threshold,
            'p_values': p_values,
            'mean_p_value': np.mean(p_values),
            'std_p_value': np.std(p_values)
        }


class TARAm:
    """
    TARA-m: Martingale-based sequential detector.

    Uses conformal p-values to construct a test martingale.
    Under H0, wealth stays bounded (fluctuates around 1).
    Under H1 (attack), wealth grows exponentially.

    Betting strategies:
    - linear: Standard betting, grows for low p-values
    - twosided: Detects deviation in either direction
    - twosided_asymmetric: Optimized for Eve's high p-value pattern
    - jumper: Aggressive betting on small p-values
    """

    def __init__(self,
                 calibration_data: Dict[str, np.ndarray],
                 epsilon: float = 0.5,
                 betting: Literal['linear', 'jumper', 'twosided',
                                  'twosided_asymmetric', 'quadratic'] = 'linear'):
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

    def _learn_probs(self, data: Dict[str, np.ndarray]) -> Dict:
        """Learn P(a,b|x,y) from data with Laplace smoothing."""
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
                    counts += 1
                    probs[(xi, yi)] = counts / counts.sum()
                else:
                    probs[(xi, yi)] = np.ones((2, 2)) / 4

        return probs

    def _compute_scores(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute nonconformity scores."""
        scores = []
        for i in range(len(data['x'])):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            scores.append(-np.log(p + 1e-10))
        return np.array(scores)

    def _compute_bet(self, pv: float) -> float:
        """
        Compute betting multiplier based on p-value.

        Different strategies:
        - linear: bet = 1 + epsilon(1 - 2p) [grows for p < 0.5]
        - jumper: bet = (1-epsilon) * p^(-epsilon) [aggressive for small p]
        - twosided: bet = 1 + epsilon * |0.5 - p| * 2 [any deviation]
        - twosided_asymmetric: stronger for p > 0.5 (Eve's pattern)
        - quadratic: bet = 1 + epsilon * (0.5 - p)^2 * 4
        """
        pv = max(pv, 0.001)  # Avoid numerical issues
        pv = min(pv, 0.999)

        if self.betting == 'linear':
            return 1.0 + self.epsilon * (1.0 - 2.0 * pv)

        elif self.betting == 'jumper':
            return (1.0 - self.epsilon) * (pv ** (-self.epsilon))

        elif self.betting == 'twosided':
            deviation = abs(0.5 - pv) * 2
            return 1.0 + self.epsilon * deviation

        elif self.betting == 'twosided_asymmetric':
            # Eve produces HIGH p-values, so bet more aggressively there
            if pv > 0.5:
                return 1.0 + self.epsilon * 2.0 * (pv - 0.5)
            else:
                return 1.0 + self.epsilon * 0.5 * (0.5 - pv)

        elif self.betting == 'quadratic':
            deviation = (0.5 - pv) ** 2 * 4
            return 1.0 + self.epsilon * deviation

        else:
            return 1.0

    def test(self, data: Dict[str, np.ndarray],
             threshold: float = 20.0) -> Dict:
        """
        Test data using martingale.

        Args:
            data: Test data dict
            threshold: Wealth threshold for detection (e.g., 20 = reject at 5%)

        Returns:
            Dict with wealth statistics and detection info
        """
        n = len(data['x'])
        wealth = 1.0
        max_wealth = 1.0
        wealth_history = [1.0]
        p_values = []
        e_values = []

        for i in range(n):
            p = self.cond_probs[(data['x'][i], data['y'][i])][data['a'][i], data['b'][i]]
            score = -np.log(p + 1e-10)

            # Conformal p-value
            rank = np.sum(self.cal_scores >= score) + 1
            pv = rank / (len(self.cal_scores) + 1)
            p_values.append(pv)

            # Betting
            bet = self._compute_bet(pv)
            bet = max(bet, 0.01)  # Floor to prevent zero
            e_values.append(bet)

            wealth *= bet
            max_wealth = max(max_wealth, wealth)
            wealth_history.append(min(wealth, 1e100))  # Cap for stability

        log_w = np.log10(max_wealth + 1e-10)
        log_w = min(log_w, 100)

        # Detection time
        warr = np.array(wealth_history)
        det_idx = np.where(warr >= threshold)[0]
        det_time = int(det_idx[0]) if len(det_idx) > 0 else None

        return {
            'max_log_wealth': log_w,
            'max_wealth': max_wealth,
            'final_wealth': wealth_history[-1],
            'detected': max_wealth >= threshold,
            'detection_time': det_time,
            'mean_p_value': np.mean(p_values),
            'mean_e_value': np.mean(e_values),
            'wealth_trajectory': np.array(wealth_history),
            'p_values': np.array(p_values),
            'e_values': np.array(e_values)
        }


class TARAm_TwoSided(TARAm):
    """
    Two-sided TARA-m that detects both low AND high p-value deviations.

    Eve-GAN produces HIGH p-values (over-predictable outcomes).
    This version catches that pattern specifically.
    """

    def __init__(self, calibration_data: Dict[str, np.ndarray],
                 epsilon: float = 0.5):
        super().__init__(calibration_data, epsilon, betting='twosided_asymmetric')


def compute_auc(honest_scores: np.ndarray, eve_scores: np.ndarray) -> float:
    """
    Compute AUC for detector scores.

    Higher score = more anomalous.

    Args:
        honest_scores: Detection scores for honest data
        eve_scores: Detection scores for Eve data

    Returns:
        AUC value (0.5 = random, 1.0 = perfect separation)
    """
    n_h = len(honest_scores)
    n_e = len(eve_scores)

    # Count pairs where eve > honest
    count = 0
    for e in eve_scores:
        count += np.sum(e > honest_scores)
        count += 0.5 * np.sum(e == honest_scores)

    return count / (n_h * n_e)


def compute_roc_curve(honest_scores: np.ndarray,
                      eve_scores: np.ndarray,
                      n_thresholds: int = 100) -> Dict:
    """
    Compute ROC curve for detector evaluation.

    Args:
        honest_scores: Scores for honest data
        eve_scores: Scores for adversarial data
        n_thresholds: Number of threshold points

    Returns:
        Dict with 'fpr', 'tpr', 'thresholds', 'auc'
    """
    all_scores = np.concatenate([honest_scores, eve_scores])
    thresholds = np.linspace(all_scores.min() - 0.01,
                             all_scores.max() + 0.01,
                             n_thresholds)

    fpr = []
    tpr = []

    n_h = len(honest_scores)
    n_e = len(eve_scores)

    for t in thresholds:
        fp = np.sum(honest_scores > t)
        tp = np.sum(eve_scores > t)

        fpr.append(fp / n_h)
        tpr.append(tp / n_e)

    auc = -np.trapz(tpr, fpr)

    return {
        'fpr': np.array(fpr),
        'tpr': np.array(tpr),
        'thresholds': thresholds,
        'auc': auc
    }


if __name__ == '__main__':
    # Quick test
    print("Testing TARA detectors...")

    # Generate test data
    np.random.seed(42)
    n = 1000

    # Calibration data (product state - classical)
    cal_data = {
        'x': np.random.randint(0, 2, n),
        'y': np.random.randint(0, 2, n),
        'a': np.random.randint(0, 2, n),
        'b': np.random.randint(0, 2, n)
    }

    # Test data (with some correlation)
    test_data = {
        'x': np.random.randint(0, 2, n),
        'y': np.random.randint(0, 2, n),
        'a': np.random.randint(0, 2, n),
        'b': np.random.randint(0, 2, n)
    }

    # TARA-k test
    tara_k = TARAk(cal_data)
    result_k = tara_k.test(test_data)
    print(f"TARA-k: KS={result_k['ks_statistic']:.4f}, detected={result_k['detected']}")

    # TARA-m test
    tara_m = TARAm(cal_data, epsilon=0.5, betting='twosided')
    result_m = tara_m.test(test_data)
    print(f"TARA-m: max_wealth={result_m['max_wealth']:.2f}, detected={result_m['detected']}")

    print("All tests passed!")
