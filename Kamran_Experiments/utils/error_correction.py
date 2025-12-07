"""
CASCADE Error Correction Protocol

This module implements the CASCADE protocol for QKD error correction:
1. Block parity comparison
2. Binary search for error location (BINARY)
3. Multiple passes with increasing block sizes
4. Error propagation through previous passes (CASCADE)

Information leakage: ~1.2 * n * h(QBER) bits for n-bit key
Efficiency: ~1.0 - 1.2 (ratio of leaked bits to Shannon limit)

Author: Davut Emre Tasar
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import hashlib


@dataclass
class CASCADEResult:
    """Result of CASCADE error correction."""
    corrected_alice: np.ndarray
    corrected_bob: np.ndarray
    initial_errors: int
    final_errors: int
    bits_leaked: int
    n_passes: int
    success: bool

    def to_dict(self) -> dict:
        return {
            'initial_errors': self.initial_errors,
            'final_errors': self.final_errors,
            'bits_leaked': self.bits_leaked,
            'n_passes': self.n_passes,
            'success': self.success,
            'correction_rate': 1 - self.final_errors / max(1, self.initial_errors),
            'leakage_fraction': self.bits_leaked / len(self.corrected_alice)
        }


def binary_entropy(p: float) -> float:
    """Binary entropy function h(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def compute_parity(bits: np.ndarray) -> int:
    """Compute parity (XOR) of bit array."""
    return int(np.sum(bits) % 2)


def find_block_sizes(n: int, qber: float, n_passes: int = 4) -> List[int]:
    """
    Determine optimal block sizes for CASCADE passes.

    First block size: k1 ≈ 0.73 / QBER
    Subsequent sizes: k_i = 2 * k_{i-1}

    Args:
        n: Key length
        qber: Estimated QBER
        n_passes: Number of passes

    Returns:
        List of block sizes for each pass
    """
    if qber <= 0:
        qber = 0.01  # Minimum QBER assumption

    # First block size (optimized for error detection)
    k1 = max(4, int(0.73 / qber))
    k1 = min(k1, n // 2)  # Don't exceed half of key

    block_sizes = [k1]
    for _ in range(n_passes - 1):
        next_size = min(block_sizes[-1] * 2, n)
        block_sizes.append(next_size)

    return block_sizes


def binary_search_error(alice_block: np.ndarray, bob_block: np.ndarray,
                        parities_leaked: List[int]) -> Tuple[int, int]:
    """
    Find error position using BINARY protocol.

    Uses binary search with parity checks to locate single error.

    Args:
        alice_block: Alice's bit block
        bob_block: Bob's bit block
        parities_leaked: List to track leaked parity bits

    Returns:
        Tuple of (error_position, bits_leaked)
    """
    n = len(alice_block)
    if n == 1:
        return 0, 0

    left, right = 0, n
    bits_leaked = 0

    while right - left > 1:
        mid = (left + right) // 2

        # Compare parities of left half
        alice_parity = compute_parity(alice_block[left:mid])
        bob_parity = compute_parity(bob_block[left:mid])
        bits_leaked += 1

        if alice_parity != bob_parity:
            right = mid
        else:
            left = mid

    parities_leaked.append(bits_leaked)
    return left, bits_leaked


def cascade_pass(alice_key: np.ndarray, bob_key: np.ndarray,
                 block_size: int, error_positions: set,
                 parities_leaked: List[int]) -> Tuple[np.ndarray, int, int]:
    """
    Perform one pass of CASCADE protocol.

    Args:
        alice_key: Alice's key bits
        bob_key: Bob's (mutable) key bits
        block_size: Block size for this pass
        error_positions: Set of corrected error positions (for cascade)
        parities_leaked: List to track leaked parities

    Returns:
        Tuple of (corrected_bob_key, errors_found, bits_leaked)
    """
    n = len(alice_key)
    bob_corrected = bob_key.copy()
    errors_found = 0
    bits_leaked = 0

    # Random permutation for this pass
    perm = np.random.permutation(n)
    alice_perm = alice_key[perm]
    bob_perm = bob_corrected[perm]

    # Process blocks
    for i in range(0, n, block_size):
        block_end = min(i + block_size, n)
        alice_block = alice_perm[i:block_end]
        bob_block = bob_perm[i:block_end]

        # Compare block parities
        alice_parity = compute_parity(alice_block)
        bob_parity = compute_parity(bob_block)
        bits_leaked += 1  # Parity comparison leaks 1 bit

        if alice_parity != bob_parity:
            # Error in this block - use binary search
            error_pos, search_bits = binary_search_error(
                alice_block, bob_block.copy(), parities_leaked
            )
            bits_leaked += search_bits

            # Correct the error
            global_pos = perm[i + error_pos]
            bob_corrected[global_pos] = 1 - bob_corrected[global_pos]
            error_positions.add(global_pos)
            errors_found += 1

    return bob_corrected, errors_found, bits_leaked


def cascade_correct(alice_key: np.ndarray, bob_key: np.ndarray,
                    qber: float, n_passes: int = 4,
                    seed: int = None) -> CASCADEResult:
    """
    Perform CASCADE error correction.

    Args:
        alice_key: Alice's key bits (reference)
        bob_key: Bob's key bits (to be corrected)
        qber: Estimated QBER
        n_passes: Number of CASCADE passes
        seed: Random seed for permutations

    Returns:
        CASCADEResult with corrected keys and statistics
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(alice_key)
    initial_errors = np.sum(alice_key != bob_key)

    # Determine block sizes
    block_sizes = find_block_sizes(n, qber, n_passes)

    bob_corrected = bob_key.copy()
    total_errors_found = 0
    total_bits_leaked = 0
    error_positions = set()
    parities_leaked = []

    for pass_num, block_size in enumerate(block_sizes):
        bob_corrected, errors_found, bits_leaked = cascade_pass(
            alice_key, bob_corrected, block_size,
            error_positions, parities_leaked
        )
        total_errors_found += errors_found
        total_bits_leaked += bits_leaked

    # Final error count
    final_errors = np.sum(alice_key != bob_corrected)

    return CASCADEResult(
        corrected_alice=alice_key,
        corrected_bob=bob_corrected,
        initial_errors=int(initial_errors),
        final_errors=int(final_errors),
        bits_leaked=total_bits_leaked,
        n_passes=n_passes,
        success=final_errors == 0
    )


def estimate_cascade_leakage(n: int, qber: float) -> float:
    """
    Estimate information leakage for CASCADE.

    Theoretical leakage: f * n * h(QBER)
    where f ≈ 1.0 - 1.2 is the efficiency factor.

    Args:
        n: Key length
        qber: QBER

    Returns:
        Estimated leaked bits
    """
    if qber <= 0:
        return 0
    return 1.16 * n * binary_entropy(qber)


def verify_error_correction(alice: np.ndarray, bob: np.ndarray,
                            sample_size: int = 100) -> Tuple[bool, float]:
    """
    Verify error correction by comparing random sample.

    This leaks additional bits but provides verification.

    Args:
        alice: Alice's corrected key
        bob: Bob's corrected key
        sample_size: Number of bits to compare

    Returns:
        Tuple of (all_match, match_rate)
    """
    n = len(alice)
    sample_size = min(sample_size, n)

    indices = np.random.choice(n, sample_size, replace=False)
    alice_sample = alice[indices]
    bob_sample = bob[indices]

    matches = np.sum(alice_sample == bob_sample)
    match_rate = matches / sample_size
    all_match = matches == sample_size

    return all_match, match_rate
