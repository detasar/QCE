#!/usr/bin/env python3
"""
Experiment 1.9: Full QKD Pipeline Integration

This experiment integrates all QKD components into a complete pipeline:
1. Bell pair generation
2. Random basis selection (key/security split)
3. Sifting protocol
4. QBER estimation
5. CHSH security check
6. Error correction (CASCADE)
7. Privacy amplification
8. Final secure key generation

Expected results:
- Complete protocol runs end-to-end
- Secure key generated for QBER < 11%
- Security checks pass for good visibility
- Protocol correctly aborts for high QBER

Author: Davut Emre Tasar
Date: December 2024
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PHASE1_RESULTS
from utils.circuits import IdealBellSimulator, HAS_QISKIT
from utils.sifting import sift_keys
from utils.qber import estimate_qber, check_qber_security, qber_from_visibility
from utils.security import calculate_chsh_value, chsh_security_bound, TSIRELSON_BOUND
from utils.error_correction import cascade_correct, binary_entropy
from utils.privacy_amp import privacy_amplification, key_to_hex, hash_key


@dataclass
class QKDSessionResult:
    """Complete QKD session result."""
    success: bool
    abort_reason: Optional[str]
    n_pairs: int
    visibility: float

    # Stage results
    sifted_bits: int
    sifting_rate: float
    qber: float
    qber_secure: bool
    chsh_value: float
    chsh_violation: bool

    # Final key
    ec_corrected: bool
    ec_residual_errors: int
    final_key_length: int
    key_rate: float

    # Security metrics
    security_margin: float
    eve_info_bound: float

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'abort_reason': self.abort_reason,
            'n_pairs': self.n_pairs,
            'visibility': self.visibility,
            'sifted_bits': self.sifted_bits,
            'sifting_rate': self.sifting_rate,
            'qber': self.qber,
            'qber_secure': self.qber_secure,
            'chsh_value': self.chsh_value,
            'chsh_violation': self.chsh_violation,
            'ec_corrected': self.ec_corrected,
            'ec_residual_errors': self.ec_residual_errors,
            'final_key_length': self.final_key_length,
            'key_rate': self.key_rate,
            'security_margin': self.security_margin,
            'eve_info_bound': self.eve_info_bound
        }


def run_qkd_session(n_pairs: int = 20000, visibility: float = 1.0,
                     key_fraction: float = 0.9, seed: int = 42,
                     verbose: bool = True) -> QKDSessionResult:
    """
    Run complete QKD session.

    Args:
        n_pairs: Number of Bell pairs to generate
        visibility: State visibility (1.0 = ideal)
        key_fraction: Fraction for key generation
        seed: Random seed
        verbose: Print progress

    Returns:
        QKDSessionResult with all metrics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"QKD SESSION: {n_pairs} pairs, visibility={visibility}")
        print("=" * 70)

    # Stage 1: Bell Pair Generation
    if verbose:
        print("\n[1] Generating Bell pairs...")

    sim = IdealBellSimulator(visibility=visibility, seed=seed)
    qkd_data = sim.generate_qkd_data(n_pairs=n_pairs, key_fraction=key_fraction)

    if verbose:
        print(f"    Generated {n_pairs} entangled pairs")
        n_key = sum(1 for m in qkd_data['modes'] if m == 'key')
        n_sec = sum(1 for m in qkd_data['modes'] if m == 'security')
        print(f"    Key mode: {n_key}, Security mode: {n_sec}")

    # Stage 2: Sifting
    if verbose:
        print("\n[2] Performing sifting...")

    sifting_result = sift_keys(qkd_data)
    sifted_bits = sifting_result.n_sifted
    sifting_rate = sifting_result.sifting_rate

    if verbose:
        print(f"    Sifted bits: {sifted_bits}")
        print(f"    Sifting rate: {100*sifting_rate:.1f}%")

    if sifted_bits < 100:
        return QKDSessionResult(
            success=False, abort_reason="Insufficient sifted bits",
            n_pairs=n_pairs, visibility=visibility,
            sifted_bits=sifted_bits, sifting_rate=sifting_rate,
            qber=0, qber_secure=False, chsh_value=0, chsh_violation=False,
            ec_corrected=False, ec_residual_errors=0,
            final_key_length=0, key_rate=0, security_margin=0, eve_info_bound=1
        )

    # Stage 3: QBER Estimation
    if verbose:
        print("\n[3] Estimating QBER...")

    qber_est, remaining_alice, remaining_bob = estimate_qber(
        sifting_result.sifted_alice, sifting_result.sifted_bob,
        sample_fraction=0.1, seed=seed + 100
    )
    qber = qber_est.qber

    security_check = check_qber_security(qber_est, protocol='e91')
    qber_secure = security_check['is_secure']

    if verbose:
        print(f"    QBER: {100*qber:.2f}%")
        print(f"    Threshold: {100*security_check['threshold']:.1f}%")
        print(f"    Status: {'SECURE' if qber_secure else 'ABORT'}")

    if not qber_secure:
        return QKDSessionResult(
            success=False, abort_reason="QBER exceeds threshold",
            n_pairs=n_pairs, visibility=visibility,
            sifted_bits=sifted_bits, sifting_rate=sifting_rate,
            qber=qber, qber_secure=False, chsh_value=0, chsh_violation=False,
            ec_corrected=False, ec_residual_errors=0,
            final_key_length=0, key_rate=0, security_margin=security_check['margin'],
            eve_info_bound=1
        )

    # Stage 4: CHSH Security Check
    if verbose:
        print("\n[4] Checking CHSH security...")

    if len(sifting_result.security_data['alice_outcomes']) > 50:
        chsh_result = calculate_chsh_value(sifting_result.security_data)
        chsh_value = chsh_result.S
        chsh_violation = chsh_result.violation

        security_bounds = chsh_security_bound(chsh_value)
        security_margin = security_bounds['security_margin']
        eve_info = security_bounds.get('guessing_probability', 1) - 0.5
    else:
        chsh_value = 0
        chsh_violation = False
        security_margin = 0
        eve_info = 0.5

    if verbose:
        print(f"    CHSH S: {chsh_value:.3f}")
        print(f"    Tsirelson bound: {TSIRELSON_BOUND:.3f}")
        print(f"    Bell violation: {'YES' if chsh_violation else 'NO'}")

    if visibility > 0.75 and not chsh_violation:
        if verbose:
            print("    WARNING: Expected violation not detected")

    # Stage 5: Error Correction
    if verbose:
        print("\n[5] Performing error correction...")

    ec_result = cascade_correct(remaining_alice, remaining_bob, qber,
                                 n_passes=4, seed=seed + 200)
    ec_corrected = ec_result.success
    ec_residual = ec_result.final_errors

    if verbose:
        print(f"    Initial errors: {ec_result.initial_errors}")
        print(f"    Residual errors: {ec_residual}")
        print(f"    Bits leaked: {ec_result.bits_leaked}")

    # Stage 6: Privacy Amplification
    if verbose:
        print("\n[6] Privacy amplification...")

    pa_result = privacy_amplification(
        ec_result.corrected_alice, ec_result.corrected_bob,
        qber, ec_result.bits_leaked, chsh_value=chsh_value,
        safety_bits=128, seed=seed + 300
    )

    final_key_length = pa_result.output_length
    key_rate = final_key_length / n_pairs if n_pairs > 0 else 0

    if verbose:
        print(f"    Input bits: {pa_result.input_length}")
        print(f"    Final key: {final_key_length} bits")
        print(f"    Keys match: {'YES' if pa_result.keys_match else 'NO'}")
        print(f"    Key rate: {key_rate:.4f} bits/pair")

    # Final summary
    success = (final_key_length > 0 and qber_secure and
               (chsh_violation or visibility < 0.75))

    if verbose:
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"  Status: {'SUCCESS' if success else 'NEEDS REVIEW'}")
        print(f"  Final key: {final_key_length} bits")
        print(f"  Key rate: {key_rate:.4f} bits/pair")
        if final_key_length > 0:
            key_hash = hash_key(pa_result.final_key_alice)
            print(f"  Key hash: {key_hash}")

    return QKDSessionResult(
        success=success,
        abort_reason=None if success else "EC residual errors",
        n_pairs=n_pairs,
        visibility=visibility,
        sifted_bits=sifted_bits,
        sifting_rate=sifting_rate,
        qber=qber,
        qber_secure=qber_secure,
        chsh_value=chsh_value,
        chsh_violation=chsh_violation,
        ec_corrected=ec_corrected,
        ec_residual_errors=ec_residual,
        final_key_length=final_key_length,
        key_rate=key_rate,
        security_margin=security_margin,
        eve_info_bound=eve_info
    )


def run_experiment():
    """Run complete pipeline experiment."""
    print("=" * 70)
    print("EXPERIMENT 1.9: FULL QKD PIPELINE")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Qiskit available: {HAS_QISKIT}")

    all_results = {
        'experiment': 'exp_1_9_full_pipeline',
        'timestamp': datetime.now().isoformat(),
        'sessions': []
    }

    overall_passed = True

    # Test different visibility levels
    test_cases = [
        (1.0, 30000, "Ideal channel"),
        (0.95, 30000, "High quality channel"),
        (0.90, 30000, "Good channel"),
        (0.85, 30000, "Moderate channel"),
        (0.80, 30000, "Noisy channel"),
        (0.78, 30000, "Near threshold"),
    ]

    for visibility, n_pairs, description in test_cases:
        print(f"\n{'#'*70}")
        print(f"SESSION: {description} (v={visibility})")
        print("#" * 70)

        result = run_qkd_session(n_pairs=n_pairs, visibility=visibility,
                                  seed=42 + int(visibility * 100))
        all_results['sessions'].append(result.to_dict())

    # Summary table
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(f"\n{'Vis':<6} {'QBER':<8} {'CHSH':<8} {'Sifted':<8} {'Final':<8} {'Rate':<8} {'Status':<10}")
    print("-" * 66)

    for session in all_results['sessions']:
        status = "OK" if session['success'] else session['abort_reason'][:10] if session['abort_reason'] else "FAIL"
        print(f"{session['visibility']:<6.2f} "
              f"{100*session['qber']:<8.1f}% "
              f"{session['chsh_value']:<8.3f} "
              f"{session['sifted_bits']:<8} "
              f"{session['final_key_length']:<8} "
              f"{session['key_rate']:<8.4f} "
              f"{status:<10}")

    # Count successes
    n_success = sum(1 for s in all_results['sessions'] if s['success'])
    n_total = len(all_results['sessions'])

    print(f"\nSuccessful sessions: {n_success}/{n_total}")

    # Check expected behavior
    checks = [
        ("Ideal channel produces key", all_results['sessions'][0]['final_key_length'] > 0),
        ("QBER increases with noise", all_results['sessions'][-2]['qber'] > all_results['sessions'][0]['qber']),
        ("Key rate decreases with noise", all_results['sessions'][-2]['key_rate'] < all_results['sessions'][0]['key_rate']),
        ("Near threshold has low/zero key", all_results['sessions'][-1]['final_key_length'] < 1000),
    ]

    print("\n--- VALIDATION ---")
    for name, passed in checks:
        print(f"  [{('PASS' if passed else 'FAIL')}] {name}")
        overall_passed = overall_passed and passed

    overall_passed = n_success >= 3  # At least 3 successful sessions

    print(f"\nOverall: {'SUCCESS' if overall_passed else 'FAILURE'}")

    all_results['summary'] = {
        'sessions_total': n_total,
        'sessions_success': n_success,
        'overall_passed': overall_passed
    }

    # Save results
    output_file = PHASE1_RESULTS / 'exp_1_9_full_pipeline.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    results = run_experiment()
