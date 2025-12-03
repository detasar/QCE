#!/usr/bin/env python3
"""
TARA Batch Detection Experiment

Demonstrates TARA-k batch anomaly detection on IBM Quantum hardware data.
This experiment validates the core TARA framework for quantum certification.

Author: Davut Emre Tasar
"""

import numpy as np
import json
import os
from typing import Dict

from data_utils import load_ibm_data, compute_chsh, split_data
from tara_detectors import TARAk, compute_auc


def run_tara_batch_experiment(
    n_runs: int = 20,
    test_size: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Run TARA-k batch detection experiment.

    Args:
        n_runs: Number of experiment runs
        test_size: Number of samples per test batch
        verbose: Print progress

    Returns:
        Dictionary with experiment results
    """
    print("=" * 70)
    print("TARA BATCH DETECTION EXPERIMENT")
    print("=" * 70)

    # Load IBM hardware data
    print("\n[1] Loading IBM Quantum hardware data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ibm_hardware_real.csv')
    ibm_data = load_ibm_data(data_path)
    n_total = len(ibm_data['x'])
    chsh = compute_chsh(ibm_data)
    print(f"    Total samples: {n_total:,}")
    print(f"    CHSH S-value: {chsh:.4f}")

    # Split into calibration and test pools
    print("\n[2] Preparing data splits...")
    np.random.seed(42)
    calib_data, test_pool = split_data(ibm_data, train_ratio=0.3, shuffle=True)
    print(f"    Calibration: {len(calib_data['x']):,} samples")
    print(f"    Test pool: {len(test_pool['x']):,} samples")

    # Initialize TARA-k detector
    print("\n[3] Initializing TARA-k detector...")
    tara_k = TARAk(calib_data)
    print("    Detector calibrated on product-state baseline")

    # Run detection experiments
    print(f"\n[4] Running {n_runs} detection trials...")

    results = {
        'ks_statistics': [],
        'p_values': [],
        'detected': [],
        'chsh_values': []
    }

    for run in range(n_runs):
        # Sample test batch
        np.random.seed(run * 100)
        idx = np.random.choice(len(test_pool['x']), test_size, replace=False)
        test_batch = {k: v[idx] for k, v in test_pool.items()}

        # Run detection
        result = tara_k.test(test_batch)
        batch_chsh = compute_chsh(test_batch)

        results['ks_statistics'].append(result['ks_statistic'])
        results['p_values'].append(result['mean_p_value'])
        results['detected'].append(result['detected'])
        results['chsh_values'].append(batch_chsh)

        if verbose and (run + 1) % 5 == 0:
            print(f"    Run {run+1}/{n_runs}: KS={result['ks_statistic']:.4f}, "
                  f"CHSH={batch_chsh:.4f}")

    # Compute summary statistics
    summary = {
        'mean_ks': float(np.mean(results['ks_statistics'])),
        'std_ks': float(np.std(results['ks_statistics'])),
        'mean_chsh': float(np.mean(results['chsh_values'])),
        'detection_rate': float(np.mean(results['detected'])),
        'n_runs': n_runs,
        'test_size': test_size,
        'calibration_size': len(calib_data['x'])
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  - Total samples: {n_total:,}")
    print(f"  - CHSH S-value: {chsh:.4f} (Tsirelson bound: 2.828)")
    print(f"\nTARA-k Detection Results:")
    print(f"  - Mean KS statistic: {summary['mean_ks']:.4f} +/- {summary['std_ks']:.4f}")
    print(f"  - Detection rate: {summary['detection_rate']:.1%}")
    print(f"  - Mean batch CHSH: {summary['mean_chsh']:.4f}")
    print(f"\nConfiguration:")
    print(f"  - Calibration samples: {summary['calibration_size']:,}")
    print(f"  - Test batch size: {test_size}")
    print(f"  - Number of runs: {n_runs}")
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        'summary': summary,
        'raw_results': {
            'ks_statistics': [float(x) for x in results['ks_statistics']],
            'chsh_values': [float(x) for x in results['chsh_values']],
            'detected': [bool(x) for x in results['detected']]
        }
    }

    output_path = os.path.join(results_dir, 'tara_batch_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='TARA Batch Detection Experiment')
    parser.add_argument('--runs', type=int, default=20, help='Number of experiment runs')
    parser.add_argument('--test-size', type=int, default=500, help='Test batch size')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')

    args = parser.parse_args()

    run_tara_batch_experiment(
        n_runs=args.runs,
        test_size=args.test_size,
        verbose=not args.quiet
    )
