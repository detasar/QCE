"""
CHSH-based Security Analysis for QKD

This module provides:
1. CHSH value calculation with statistical error
2. Device-independent security bounds
3. Key rate from Bell violation
4. Eve information estimation

The CHSH inequality S <= 2 is violated by quantum mechanics:
- Maximum quantum value: 2*sqrt(2) ≈ 2.828 (Tsirelson bound)
- Classical bound: 2.0
- Violation indicates genuine quantum correlations

For device-independent QKD:
- CHSH > 2 → Some secrecy is possible
- Higher CHSH → More secure key
- CHSH < 2 → Abort (possible classical simulation)

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from scipy import stats


# Physical constants
TSIRELSON_BOUND = 2 * np.sqrt(2)  # ≈ 2.828
CLASSICAL_BOUND = 2.0


@dataclass
class CHSHResult:
    """Result of CHSH calculation."""
    S: float                 # CHSH value
    S_std: float            # Standard error
    correlators: Dict[str, float]  # E(a,b) values
    correlator_stds: Dict[str, float]  # Standard errors
    counts: Dict[str, int]  # Sample counts per setting
    total_samples: int      # Total security samples
    violation: bool         # S > 2?
    sigma_violation: float  # Significance in sigmas

    def to_dict(self) -> dict:
        return {
            'S': self.S,
            'S_std': self.S_std,
            'correlators': self.correlators,
            'correlator_stds': self.correlator_stds,
            'counts': self.counts,
            'total_samples': self.total_samples,
            'violation': self.violation,
            'sigma_violation': self.sigma_violation
        }


def calculate_chsh_value(security_data: dict) -> CHSHResult:
    """
    Calculate CHSH S value from security test data.

    CHSH formula:
        S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)

    where E(a,b) = P(same) - P(different)

    For optimal angles (0, 45, 22.5, 67.5 degrees):
        |S_max| = 2*sqrt(2) ≈ 2.828

    Args:
        security_data: Dictionary with:
            - alice_settings: list of 0/1 for Alice's setting
            - bob_settings: list of 0/1 for Bob's setting
            - alice_outcomes: list of 0/1 for Alice's outcome
            - bob_outcomes: list of 0/1 for Bob's outcome

    Returns:
        CHSHResult with S value and statistics
    """
    alice_settings = np.array(security_data['alice_settings'])
    bob_settings = np.array(security_data['bob_settings'])
    alice_outcomes = np.array(security_data['alice_outcomes'])
    bob_outcomes = np.array(security_data['bob_outcomes'])

    correlators = {}
    correlator_stds = {}
    counts = {}

    # Calculate correlator for each setting combination
    for a_set in [0, 1]:
        for b_set in [0, 1]:
            key = f"({a_set}, {b_set})"

            # Find indices with this setting
            mask = (alice_settings == a_set) & (bob_settings == b_set)
            n = np.sum(mask)

            if n == 0:
                correlators[key] = 0.0
                correlator_stds[key] = 0.0
                counts[key] = 0
                continue

            a_vals = alice_outcomes[mask]
            b_vals = bob_outcomes[mask]

            # E = P(same) - P(different) = 2*P(same) - 1
            same = np.sum(a_vals == b_vals)
            E = (2 * same - n) / n

            # Standard error: Var(E) ≈ (1 - E^2) / n
            var_E = (1 - E**2) / n
            std_E = np.sqrt(var_E)

            correlators[key] = float(E)
            correlator_stds[key] = float(std_E)
            counts[key] = int(n)

    # CHSH formula: S = E(0,0) - E(0,1) + E(1,0) + E(1,1)
    S = (correlators['(0, 0)'] - correlators['(0, 1)'] +
         correlators['(1, 0)'] + correlators['(1, 1)'])

    # Error propagation: sigma_S = sqrt(sum of sigma_i^2)
    S_std = np.sqrt(sum(s**2 for s in correlator_stds.values()))

    # Significance of violation
    violation = S > CLASSICAL_BOUND
    sigma_violation = (S - CLASSICAL_BOUND) / S_std if S_std > 0 else 0

    return CHSHResult(
        S=S,
        S_std=S_std,
        correlators=correlators,
        correlator_stds=correlator_stds,
        counts=counts,
        total_samples=len(alice_settings),
        violation=violation,
        sigma_violation=sigma_violation
    )


def chsh_security_bound(S: float) -> dict:
    """
    Calculate device-independent security bounds from CHSH value.

    For S > 2, the min-entropy is bounded from below:
        H_min(A|E) >= 1 - log2(1 + sqrt(2 - (S/2)^2) / 2)

    The key rate is:
        r = H_min(A|E) - H(A|B)

    Args:
        S: CHSH value

    Returns:
        Dictionary with security bounds
    """
    if S <= CLASSICAL_BOUND:
        return {
            'violation': False,
            'min_entropy': 0,
            'guessing_probability': 1.0,
            'key_rate_bound': 0,
            'security_margin': 0
        }

    # Min-entropy bound (simplified Pironio-Acin-Brunner)
    # This is the local bound; DI-QKD requires entropy accumulation
    x = S / 2
    if x > 1:
        x = min(x, np.sqrt(2))  # Can't exceed Tsirelson
        term = np.sqrt(x**2 - 1)
        p_guess = (1 + np.sqrt(2 - x**2)) / 2
        p_guess = min(1.0, max(0.5, p_guess))
    else:
        p_guess = 0.5

    min_entropy = -np.log2(p_guess) if p_guess > 0 else 0

    # Key rate assuming zero QBER (need QBER correction in practice)
    key_rate = max(0, 1 - np.log2(1 + 2 * p_guess - 1)) if p_guess < 1 else 0

    # Security margin (how far above classical bound)
    margin = S - CLASSICAL_BOUND
    margin_percent = 100 * margin / (TSIRELSON_BOUND - CLASSICAL_BOUND)

    return {
        'violation': True,
        'min_entropy': min_entropy,
        'guessing_probability': p_guess,
        'key_rate_bound': key_rate,
        'security_margin': margin,
        'security_margin_percent': margin_percent,
        'S': S,
        'S_max': TSIRELSON_BOUND,
        'S_classical': CLASSICAL_BOUND
    }


def eve_information_bound(S: float, qber: float = 0) -> dict:
    """
    Estimate Eve's information from CHSH value.

    For CHSH value S, Eve's information is bounded:
        I(A:E) <= 1 - h(p_success)

    where p_success depends on S through the tilted CHSH game.

    Args:
        S: CHSH value
        qber: Quantum bit error rate

    Returns:
        Dictionary with Eve information bounds
    """
    if S <= CLASSICAL_BOUND:
        # No violation → Eve could have full information
        return {
            'eve_info_upper': 1.0,
            'eve_info_lower': 0.5,
            'secure': False
        }

    # Simplified bound: larger S → less Eve information
    # This is a rough approximation; real DI-QKD uses entropy accumulation
    s_normalized = (S - CLASSICAL_BOUND) / (TSIRELSON_BOUND - CLASSICAL_BOUND)
    s_normalized = min(1.0, max(0.0, s_normalized))

    # Eve's info decreases as S increases
    eve_info_upper = 0.5 * (1 - s_normalized)

    # Account for QBER
    if qber > 0:
        eve_info_upper += qber  # Rough correction

    eve_info_upper = min(1.0, eve_info_upper)

    return {
        'eve_info_upper': eve_info_upper,
        'eve_info_lower': 0.0,  # Could be 0 if quantum state is pure
        'secure': eve_info_upper < 0.5,
        'S_normalized': s_normalized
    }


def minimum_chsh_for_security(target_key_rate: float = 0.1) -> float:
    """
    Calculate minimum CHSH value needed for target key rate.

    Args:
        target_key_rate: Desired key rate per raw bit

    Returns:
        Minimum CHSH value required
    """
    # Numerical search (could be made analytical)
    for s in np.linspace(2.0, TSIRELSON_BOUND, 1000):
        bounds = chsh_security_bound(s)
        if bounds['key_rate_bound'] >= target_key_rate:
            return s
    return TSIRELSON_BOUND


def visibility_for_chsh(target_S: float) -> float:
    """
    Calculate required visibility for target CHSH value.

    For Bell state with visibility v:
        S_expected = 2*sqrt(2) * v

    Args:
        target_S: Target CHSH value

    Returns:
        Required visibility
    """
    return target_S / TSIRELSON_BOUND


def expected_chsh_from_visibility(visibility: float) -> float:
    """
    Calculate expected CHSH value from visibility.

    Args:
        visibility: State visibility (0 to 1)

    Returns:
        Expected CHSH value
    """
    return TSIRELSON_BOUND * visibility


def analyze_chsh_statistics(chsh_result: CHSHResult) -> dict:
    """
    Perform detailed statistical analysis of CHSH result.

    Args:
        chsh_result: CHSH calculation result

    Returns:
        Dictionary with statistical analysis
    """
    S = chsh_result.S
    S_std = chsh_result.S_std

    # Hypothesis test: H0: S <= 2 (classical)
    z_score = (S - CLASSICAL_BOUND) / S_std if S_std > 0 else 0
    p_value = 1 - stats.norm.cdf(z_score)

    # Confidence interval for S
    z_95 = 1.96
    ci_lower = S - z_95 * S_std
    ci_upper = S + z_95 * S_std

    # Check if CI excludes classical bound
    excludes_classical = ci_lower > CLASSICAL_BOUND

    # Estimate visibility from S
    estimated_visibility = S / TSIRELSON_BOUND

    return {
        'S': S,
        'S_std': S_std,
        'z_score': z_score,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'excludes_classical': excludes_classical,
        'estimated_visibility': estimated_visibility,
        'tsirelson_efficiency': S / TSIRELSON_BOUND
    }
