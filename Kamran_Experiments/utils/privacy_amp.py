"""
Privacy Amplification using Toeplitz Hashing

This module implements privacy amplification for QKD:
1. Generate random Toeplitz matrix
2. Compress key to remove Eve's information
3. Output length based on security analysis

The final key length is:
    m = n - leak_EC - chi(QBER, S) - safety_margin

where chi is Eve's max information from QBER and CHSH value.

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import hashlib


@dataclass
class PrivacyAmpResult:
    """Result of privacy amplification."""
    input_length: int       # Input key length
    output_length: int      # Output key length
    compression_ratio: float  # output/input
    final_key_alice: np.ndarray
    final_key_bob: np.ndarray
    keys_match: bool
    security_parameter: float  # 2^(-t) failure probability

    def to_dict(self) -> dict:
        return {
            'input_length': self.input_length,
            'output_length': self.output_length,
            'compression_ratio': self.compression_ratio,
            'keys_match': self.keys_match,
            'security_parameter': self.security_parameter
        }


def binary_entropy(p: float) -> float:
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def generate_toeplitz_matrix(n_rows: int, n_cols: int,
                              seed: int = None) -> np.ndarray:
    """
    Generate a random Toeplitz matrix for privacy amplification.

    A Toeplitz matrix is constant along diagonals:
        T[i,j] = T[i+1,j+1]

    Requires only n_rows + n_cols - 1 random bits.

    Args:
        n_rows: Output dimension (final key length)
        n_cols: Input dimension (corrected key length)
        seed: Random seed

    Returns:
        Binary Toeplitz matrix of shape (n_rows, n_cols)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate the first row and first column
    first_row = np.random.randint(0, 2, n_cols)
    first_col = np.random.randint(0, 2, n_rows)
    first_col[0] = first_row[0]  # Ensure consistency

    # Build Toeplitz matrix
    T = np.zeros((n_rows, n_cols), dtype=np.int8)
    for i in range(n_rows):
        for j in range(n_cols):
            if j >= i:
                T[i, j] = first_row[j - i]
            else:
                T[i, j] = first_col[i - j]

    return T


def toeplitz_hash(key: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply Toeplitz hash to compress key.

    Args:
        key: Input key (binary array)
        T: Toeplitz matrix

    Returns:
        Compressed key (binary array)
    """
    # Matrix-vector multiplication mod 2
    result = np.dot(T, key) % 2
    return result.astype(np.int8)


def calculate_final_key_length(n_corrected: int, qber: float,
                                 leak_ec: int, chsh_value: float = None,
                                 safety_bits: int = 128) -> int:
    """
    Calculate secure final key length.

    Using Devetak-Winter bound for collective attacks:
        r = 1 - h(QBER) - h(QBER)  (for BB84-type)

    For DI-QKD with CHSH:
        r = 1 - h(p_guess(S))

    Args:
        n_corrected: Number of corrected bits
        qber: Estimated QBER
        leak_ec: Bits leaked during error correction
        chsh_value: CHSH value (for DI bound)
        safety_bits: Additional safety margin

    Returns:
        Secure final key length
    """
    # Eve's information from QBER (prepare-and-measure bound)
    chi_pm = n_corrected * binary_entropy(qber)

    # If CHSH available, use DI bound
    if chsh_value is not None and chsh_value > 2.0:
        # Simplified DI bound
        s_norm = (chsh_value - 2.0) / (2.828 - 2.0)
        chi_di = n_corrected * (1 - s_norm) * 0.5  # Rough approximation
        chi = min(chi_pm, chi_di)
    else:
        chi = chi_pm

    # Final key length
    m = n_corrected - leak_ec - int(np.ceil(chi)) - safety_bits

    return max(0, int(m))


def privacy_amplification(alice_key: np.ndarray, bob_key: np.ndarray,
                           qber: float, leak_ec: int,
                           chsh_value: float = None,
                           safety_bits: int = 128,
                           seed: int = None) -> PrivacyAmpResult:
    """
    Perform privacy amplification.

    Args:
        alice_key: Alice's corrected key
        bob_key: Bob's corrected key
        qber: Estimated QBER
        leak_ec: Bits leaked during error correction
        chsh_value: CHSH value (optional, for DI bound)
        safety_bits: Security parameter (2^(-t) failure probability)
        seed: Random seed for Toeplitz matrix

    Returns:
        PrivacyAmpResult with final keys
    """
    n = len(alice_key)

    # Calculate final key length
    m = calculate_final_key_length(n, qber, leak_ec, chsh_value, safety_bits)

    if m <= 0:
        # No secure key possible
        return PrivacyAmpResult(
            input_length=n,
            output_length=0,
            compression_ratio=0,
            final_key_alice=np.array([], dtype=np.int8),
            final_key_bob=np.array([], dtype=np.int8),
            keys_match=True,  # Vacuously true
            security_parameter=safety_bits
        )

    # Generate Toeplitz matrix (shared via authenticated channel)
    T = generate_toeplitz_matrix(m, n, seed=seed)

    # Apply to both keys
    final_alice = toeplitz_hash(alice_key, T)
    final_bob = toeplitz_hash(bob_key, T)

    # Check if keys match (they should if EC was successful)
    keys_match = np.array_equal(final_alice, final_bob)

    return PrivacyAmpResult(
        input_length=n,
        output_length=m,
        compression_ratio=m / n if n > 0 else 0,
        final_key_alice=final_alice,
        final_key_bob=final_bob,
        keys_match=keys_match,
        security_parameter=safety_bits
    )


def key_to_hex(key: np.ndarray) -> str:
    """Convert binary key to hexadecimal string."""
    if len(key) == 0:
        return ""

    # Pad to multiple of 8
    padded_len = ((len(key) + 7) // 8) * 8
    padded = np.zeros(padded_len, dtype=np.int8)
    padded[:len(key)] = key

    # Convert to bytes
    bytes_list = []
    for i in range(0, padded_len, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | padded[i + j]
        bytes_list.append(byte_val)

    return bytes(bytes_list).hex()


def hash_key(key: np.ndarray, algorithm: str = 'sha256') -> str:
    """Hash key for verification (without revealing key)."""
    key_bytes = key_to_hex(key)
    if algorithm == 'sha256':
        return hashlib.sha256(key_bytes.encode()).hexdigest()[:16]
    return key_bytes[:16]


def estimate_key_rate(n_pairs: int, sifting_rate: float, qber: float,
                       ec_efficiency: float = 1.16) -> dict:
    """
    Estimate final key rate for QKD session.

    Args:
        n_pairs: Number of Bell pairs generated
        sifting_rate: Fraction of matching bases
        qber: Quantum bit error rate
        ec_efficiency: CASCADE efficiency factor

    Returns:
        Key rate estimates
    """
    # Sifted key
    n_sifted = n_pairs * sifting_rate

    # After error correction
    leak_ec = ec_efficiency * n_sifted * binary_entropy(qber)

    # Eve's information
    chi = n_sifted * binary_entropy(qber)

    # Final key (asymptotic)
    r = 1 - 2 * binary_entropy(qber)  # Shor-Preskill bound
    n_final = max(0, n_sifted * r - 128)  # With safety margin

    return {
        'n_pairs': n_pairs,
        'n_sifted': n_sifted,
        'qber': qber,
        'sifting_rate': sifting_rate,
        'key_rate': r,
        'final_key_length': int(n_final),
        'efficiency': n_final / n_pairs if n_pairs > 0 else 0
    }
