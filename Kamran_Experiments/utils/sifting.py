"""
Key Sifting Protocol for QKD

This module implements the sifting phase of QKD protocols:
1. Basis reconciliation over authenticated channel
2. Discarding non-matching basis measurements
3. Separation of key bits from security test bits

The sifting rate depends on basis choice distribution:
- Uniform (50% Z, 50% X): Expected ~50% sifting rate
- Optimal for QKD depends on security vs efficiency tradeoff

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


class BasisType(Enum):
    """Measurement basis types."""
    Z = "Z"      # Computational basis (0/1)
    X = "X"      # Hadamard basis (+/-)
    CHSH = "CHSH"  # Security test angle


@dataclass
class SiftingResult:
    """Result of key sifting protocol."""
    sifted_alice: np.ndarray   # Alice's sifted key bits
    sifted_bob: np.ndarray     # Bob's sifted key bits
    sifted_indices: List[int]  # Original indices of sifted bits
    sifting_rate: float        # Fraction of matching bases
    n_total_key: int           # Total key mode samples
    n_sifted: int              # Number of sifted bits
    security_data: Dict        # Data for CHSH calculation

    def to_dict(self) -> dict:
        return {
            'sifted_alice': self.sifted_alice.tolist(),
            'sifted_bob': self.sifted_bob.tolist(),
            'sifted_indices': self.sifted_indices,
            'sifting_rate': self.sifting_rate,
            'n_total_key': self.n_total_key,
            'n_sifted': self.n_sifted,
            'security_samples': len(self.security_data.get('settings', []))
        }


def sift_keys(qkd_data: dict) -> SiftingResult:
    """
    Perform key sifting based on basis comparison.

    In a real QKD protocol:
    1. Alice announces her bases over authenticated channel
    2. Bob compares with his bases
    3. They keep only matching basis measurements
    4. Security test data is kept separately for CHSH

    Args:
        qkd_data: Dictionary with:
            - modes: List of 'key' or 'security'
            - alice_bases: Alice's basis choices
            - bob_bases: Bob's basis choices
            - alice_bits: Alice's measurement outcomes
            - bob_bits: Bob's measurement outcomes
            - alice_settings: CHSH settings (for security)
            - bob_settings: CHSH settings (for security)

    Returns:
        SiftingResult with sifted keys and security data
    """
    modes = qkd_data['modes']
    alice_bases = qkd_data['alice_bases']
    bob_bases = qkd_data['bob_bases']
    alice_bits = np.array(qkd_data['alice_bits'])
    bob_bits = np.array(qkd_data['bob_bits'])

    sifted_alice = []
    sifted_bob = []
    sifted_indices = []

    security_data = {
        'settings': [],
        'alice_outcomes': [],
        'bob_outcomes': [],
        'alice_settings': [],
        'bob_settings': []
    }

    n_key_mode = 0

    for i in range(len(modes)):
        if modes[i] == 'key':
            n_key_mode += 1

            # For key generation: keep only matching bases
            if alice_bases[i] == bob_bases[i]:
                sifted_alice.append(alice_bits[i])
                sifted_bob.append(bob_bits[i])
                sifted_indices.append(i)

        else:  # Security test mode
            # Keep all security test data for CHSH
            security_data['settings'].append((alice_bases[i], bob_bases[i]))
            security_data['alice_outcomes'].append(int(alice_bits[i]))
            security_data['bob_outcomes'].append(int(bob_bits[i]))
            security_data['alice_settings'].append(int(qkd_data['alice_settings'][i]))
            security_data['bob_settings'].append(int(qkd_data['bob_settings'][i]))

    sifted_alice = np.array(sifted_alice)
    sifted_bob = np.array(sifted_bob)

    sifting_rate = len(sifted_alice) / n_key_mode if n_key_mode > 0 else 0

    return SiftingResult(
        sifted_alice=sifted_alice,
        sifted_bob=sifted_bob,
        sifted_indices=sifted_indices,
        sifting_rate=sifting_rate,
        n_total_key=n_key_mode,
        n_sifted=len(sifted_alice),
        security_data=security_data
    )


def calculate_chsh_from_security_data(security_data: dict) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Calculate CHSH value from security test samples.

    Uses the standard CHSH formula:
        S = E(a1,b1) - E(a1,b2) + E(a2,b1) + E(a2,b2)

    Where E(a,b) = P(same) - P(different) for setting (a,b)

    Args:
        security_data: Dictionary with settings and outcomes

    Returns:
        Tuple of (S value, correlator dict, sample counts)
    """
    settings = security_data['settings']
    alice_outcomes = np.array(security_data['alice_outcomes'])
    bob_outcomes = np.array(security_data['bob_outcomes'])
    alice_settings = np.array(security_data['alice_settings'])
    bob_settings = np.array(security_data['bob_settings'])

    correlators = {}
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
                counts[key] = 0
                continue

            a_vals = alice_outcomes[mask]
            b_vals = bob_outcomes[mask]

            # E = P(same) - P(different)
            same = np.sum(a_vals == b_vals)
            diff = n - same
            correlators[key] = (same - diff) / n
            counts[key] = int(n)

    # CHSH formula: S = E(0,0) - E(0,1) + E(1,0) + E(1,1)
    S = (correlators['(0, 0)'] - correlators['(0, 1)'] +
         correlators['(1, 0)'] + correlators['(1, 1)'])

    return S, correlators, counts


def analyze_sifting_statistics(qkd_data: dict) -> dict:
    """
    Analyze detailed sifting statistics.

    Args:
        qkd_data: QKD measurement data

    Returns:
        Dictionary with detailed statistics
    """
    modes = qkd_data['modes']
    alice_bases = qkd_data['alice_bases']
    bob_bases = qkd_data['bob_bases']

    n_total = len(modes)
    n_key = sum(1 for m in modes if m == 'key')
    n_security = sum(1 for m in modes if m == 'security')

    # Count basis combinations for key mode
    key_basis_counts = {
        'ZZ': 0, 'ZX': 0, 'XZ': 0, 'XX': 0
    }

    for i in range(n_total):
        if modes[i] == 'key':
            combo = f"{alice_bases[i]}{bob_bases[i]}"
            if combo in key_basis_counts:
                key_basis_counts[combo] += 1

    # Matching = ZZ + XX, Non-matching = ZX + XZ
    n_matching = key_basis_counts['ZZ'] + key_basis_counts['XX']
    n_non_matching = key_basis_counts['ZX'] + key_basis_counts['XZ']

    # Expected sifting rate for uniform basis choice: 50%
    # For 50% Z, 50% X each: P(match) = P(ZZ) + P(XX) = 0.25 + 0.25 = 0.5
    expected_sifting = 0.5
    actual_sifting = n_matching / n_key if n_key > 0 else 0

    return {
        'total_samples': n_total,
        'key_samples': n_key,
        'security_samples': n_security,
        'key_fraction': n_key / n_total if n_total > 0 else 0,
        'basis_counts': key_basis_counts,
        'n_matching': n_matching,
        'n_non_matching': n_non_matching,
        'actual_sifting_rate': actual_sifting,
        'expected_sifting_rate': expected_sifting,
        'sifting_efficiency': actual_sifting / expected_sifting if expected_sifting > 0 else 0
    }


def verify_sifted_correlation(sifting_result: SiftingResult,
                               expected_qber: float = 0.0,
                               tolerance: float = 0.02) -> dict:
    """
    Verify that sifted keys have expected correlation.

    For ideal Bell state with matching bases:
    - Perfect visibility: QBER = 0
    - Visibility v: QBER ≈ (1-v)/2

    Args:
        sifting_result: Result from sift_keys
        expected_qber: Expected error rate
        tolerance: Acceptable deviation

    Returns:
        Verification result dictionary
    """
    if sifting_result.n_sifted == 0:
        return {'valid': False, 'error': 'No sifted bits'}

    # Calculate actual QBER
    errors = np.sum(sifting_result.sifted_alice != sifting_result.sifted_bob)
    actual_qber = errors / sifting_result.n_sifted

    # Check if within tolerance
    is_valid = abs(actual_qber - expected_qber) <= tolerance

    return {
        'valid': is_valid,
        'actual_qber': actual_qber,
        'expected_qber': expected_qber,
        'n_errors': int(errors),
        'n_sifted': sifting_result.n_sifted,
        'tolerance': tolerance
    }
