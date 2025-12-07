"""
QBER (Quantum Bit Error Rate) Estimation

This module provides:
1. QBER estimation from sifted keys
2. Confidence interval calculation
3. Sample size determination
4. Security threshold checking

The QBER is crucial for QKD security:
- E91/BBM92: QBER < 11% for information-theoretic security
- Device-independent: QBER < 7.1% from CHSH bound
- QBER > threshold → abort protocol (Eve detected)

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import stats


# Security thresholds for different protocols
QBER_THRESHOLDS = {
    'bb84': 0.11,           # 11% - standard BB84 threshold
    'e91': 0.11,            # 11% - E91/BBM92 entanglement-based
    'device_independent': 0.071,  # 7.1% - DI-QKD from CHSH
    'practical': 0.08,      # 8% - practical systems with margin
}


@dataclass
class QBEREstimate:
    """Result of QBER estimation."""
    qber: float                # Point estimate
    ci_lower: float            # Lower confidence bound
    ci_upper: float            # Upper confidence bound
    ci_width: float            # Half-width of CI
    n_sample: int              # Bits used for estimation
    n_errors: int              # Number of errors found
    confidence: float          # Confidence level
    remaining_bits: int        # Bits remaining for key

    def to_dict(self) -> dict:
        return {
            'qber': self.qber,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_width': self.ci_width,
            'n_sample': self.n_sample,
            'n_errors': self.n_errors,
            'confidence': self.confidence,
            'remaining_bits': self.remaining_bits
        }


def required_qber_samples(precision: float = 0.01,
                          confidence: float = 0.99) -> int:
    """
    Calculate required sample size for QBER estimation.

    Uses Hoeffding inequality:
        P(|QBER_est - QBER_true| > epsilon) <= 2*exp(-2*n*epsilon^2)

    For confidence level 1-delta:
        n >= ln(2/delta) / (2*epsilon^2)

    Args:
        precision: Desired precision (epsilon)
        confidence: Desired confidence level (1-delta)

    Returns:
        Required number of samples

    Examples:
        1% precision, 99% confidence → 26,492 samples
        2% precision, 95% confidence → 4,612 samples
        5% precision, 90% confidence → 461 samples
    """
    delta = 1 - confidence
    n = np.log(2 / delta) / (2 * precision ** 2)
    return int(np.ceil(n))


def estimate_qber(sifted_alice: np.ndarray,
                  sifted_bob: np.ndarray,
                  sample_fraction: float = 0.15,
                  confidence: float = 0.95,
                  min_samples: int = 100,
                  max_samples: int = 5000,
                  seed: int = None) -> Tuple[QBEREstimate, np.ndarray, np.ndarray]:
    """
    Estimate QBER with confidence interval.

    Randomly samples a fraction of sifted bits for QBER estimation.
    Returns remaining bits for key generation.

    Args:
        sifted_alice: Alice's sifted key bits
        sifted_bob: Bob's sifted key bits
        sample_fraction: Fraction of bits for estimation
        confidence: Confidence level for interval
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (QBEREstimate, remaining_alice, remaining_bob)
    """
    rng = np.random.RandomState(seed)

    n = len(sifted_alice)
    n_sample = max(min_samples, min(int(sample_fraction * n), max_samples))
    n_sample = min(n_sample, n)  # Can't sample more than we have

    if n_sample == 0:
        raise ValueError("No bits available for QBER estimation")

    # Random sample selection
    all_indices = np.arange(n)
    sample_indices = rng.choice(n, n_sample, replace=False)
    remaining_indices = np.setdiff1d(all_indices, sample_indices)

    # Count errors in sample
    sample_alice = sifted_alice[sample_indices]
    sample_bob = sifted_bob[sample_indices]
    n_errors = int(np.sum(sample_alice != sample_bob))
    qber = n_errors / n_sample

    # Wilson score confidence interval
    # More accurate than normal approximation for small n or extreme p
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z ** 2 / n_sample

    center = (qber + z ** 2 / (2 * n_sample)) / denominator
    spread = z * np.sqrt((qber * (1 - qber) + z ** 2 / (4 * n_sample)) / n_sample) / denominator

    ci_lower = max(0, center - spread)
    ci_upper = min(1, center + spread)

    estimate = QBEREstimate(
        qber=qber,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=spread,
        n_sample=n_sample,
        n_errors=n_errors,
        confidence=confidence,
        remaining_bits=len(remaining_indices)
    )

    remaining_alice = sifted_alice[remaining_indices]
    remaining_bob = sifted_bob[remaining_indices]

    return estimate, remaining_alice, remaining_bob


def check_qber_security(qber_estimate: QBEREstimate,
                        protocol: str = 'e91') -> dict:
    """
    Check if QBER is within security threshold.

    Args:
        qber_estimate: QBER estimation result
        protocol: Protocol type ('bb84', 'e91', 'device_independent', 'practical')

    Returns:
        Security check result
    """
    threshold = QBER_THRESHOLDS.get(protocol, 0.11)

    # Use upper CI bound for conservative security check
    is_secure = qber_estimate.ci_upper < threshold

    # Calculate security margin
    margin = threshold - qber_estimate.ci_upper
    margin_percent = 100 * margin / threshold if threshold > 0 else 0

    return {
        'protocol': protocol,
        'threshold': threshold,
        'qber': qber_estimate.qber,
        'qber_upper_bound': qber_estimate.ci_upper,
        'is_secure': is_secure,
        'margin': margin,
        'margin_percent': margin_percent,
        'recommendation': 'PROCEED' if is_secure else 'ABORT'
    }


def qber_from_visibility(visibility: float) -> float:
    """
    Calculate expected QBER from Bell state visibility.

    For |Phi+> with visibility v:
        P(error|matching_basis) = (1-v)/2

    Args:
        visibility: State visibility (0 to 1)

    Returns:
        Expected QBER
    """
    return (1 - visibility) / 2


def visibility_from_qber(qber: float) -> float:
    """
    Estimate visibility from measured QBER.

    Args:
        qber: Measured QBER

    Returns:
        Estimated visibility
    """
    return 1 - 2 * qber


def secrecy_capacity_bb84(qber: float) -> float:
    """
    Calculate asymptotic secret key rate for BB84-type protocols.

    Using Shor-Preskill bound:
        r = 1 - 2*h(QBER)

    where h(x) = -x*log2(x) - (1-x)*log2(1-x) is binary entropy.

    Args:
        qber: Quantum bit error rate

    Returns:
        Secret key rate (per raw bit)
    """
    if qber == 0:
        return 1.0
    if qber >= 0.5:
        return 0.0

    # Binary entropy
    h_qber = -qber * np.log2(qber) - (1 - qber) * np.log2(1 - qber)

    rate = 1 - 2 * h_qber
    return max(0, rate)


def secrecy_capacity_device_independent(chsh_value: float) -> float:
    """
    Calculate asymptotic DI secret key rate from CHSH value.

    Using the Pironio-Acin-Brunner bound (simplified):
        r = 1 - h(p_guess)

    where p_guess is related to CHSH via:
        p_guess = (1 + sqrt((S/2)^2 - 1)) / 2

    Args:
        chsh_value: CHSH S value

    Returns:
        Secret key rate per round
    """
    if chsh_value <= 2.0:
        return 0.0  # No violation = no security

    # Compute guessing probability
    s_over_2 = chsh_value / 2
    if s_over_2 >= 1:
        # Maximum value
        p_guess = (1 + np.sqrt(s_over_2 ** 2 - 1)) / 2
    else:
        p_guess = 0.5

    # Limit to valid range
    p_guess = min(1.0, max(0.5, p_guess))

    if p_guess == 1:
        return 0.0
    if p_guess == 0.5:
        return 0.0

    # Binary entropy
    h_guess = -p_guess * np.log2(p_guess) - (1 - p_guess) * np.log2(1 - p_guess)

    rate = 1 - h_guess
    return max(0, rate)
