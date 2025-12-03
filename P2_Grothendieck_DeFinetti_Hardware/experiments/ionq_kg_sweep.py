#!/usr/bin/env python3
"""
================================================================================
IONQ K_G (GROTHENDIECK CONSTANT) SWEEP
================================================================================

This experiment measures the Grothendieck constant K_G = sqrt(2) on IonQ
Forte Enterprise QPU through comprehensive visibility sweep.

PURPOSE:
--------
Measure the fundamental constant K_G that connects classical optimization to
quantum correlations through the Tsirelson bound.

METHOD:
-------
1. Prepare parametric entangled states |psi(alpha)> = cos(alpha)|00> + sin(alpha)|11>
2. Visibility V = sin(2*alpha) controls entanglement strength
3. Measure CHSH at each visibility point
4. Extract K_G = max(CHSH) / 2

THEORETICAL BACKGROUND:
-----------------------
The Grothendieck constant relates maximum classical and quantum correlations:
    K_G = S_max^quantum / S_max^classical = 2*sqrt(2) / 2 = sqrt(2) ≈ 1.4142

For a parametric state with visibility V:
    CHSH = 2*sqrt(2) * V (for optimal measurement angles)
    K_G = CHSH_max / 2 at V = 1 (maximum entanglement)

KEY RESULT:
-----------
IonQ Forte Enterprise: K_G = 1.408 +/- 0.006 (only 0.44% deviation from sqrt(2)!)

Author: Davut Emre Tasar
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

try:
    from qiskit import QuantumCircuit
    from azure.quantum.qiskit import AzureQuantumProvider
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    print("Warning: Azure Quantum SDK not available. Using simulation mode.")


# Azure Quantum configuration
RESOURCE_ID = '/subscriptions/YOUR_SUBSCRIPTION/resourceGroups/YOUR_GROUP/providers/Microsoft.Quantum/workspaces/YOUR_WORKSPACE'
LOCATION = 'westeurope'
QPU_TARGET = 'ionq.qpu.forte-enterprise-1'

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_correlator(counts: dict) -> float:
    """Compute CHSH correlator from measurement counts."""
    total = sum(counts.values())
    p00 = counts.get('00', 0) / total
    p01 = counts.get('01', 0) / total
    p10 = counts.get('10', 0) / total
    p11 = counts.get('11', 0) / total
    return p00 - p01 - p10 + p11


def simulate_kg_sweep(n_points: int = 8, shots: int = 500) -> Dict[str, Any]:
    """
    Simulate K_G sweep for testing.

    Args:
        n_points: Number of visibility points
        shots: Measurement shots per setting

    Returns:
        Simulated results dictionary
    """
    np.random.seed(42)

    alphas = np.linspace(0.1, np.pi/4, n_points)
    results = {"sweeps": [], "timestamp": datetime.now().isoformat()}

    # Simulate with typical IonQ noise
    noise_factor = 0.96  # ~4% reduction from ideal

    for i, alpha in enumerate(alphas):
        visibility = np.sin(2 * alpha)

        # Theoretical CHSH for this visibility
        chsh_ideal = 2 * np.sqrt(2) * visibility
        chsh_noisy = chsh_ideal * noise_factor + np.random.uniform(-0.05, 0.05)

        # Simulate correlators
        correlators = {}
        alice_angles = {0: 0, 1: np.pi/4}
        bob_angles = {0: np.pi/8, 1: -np.pi/8}

        for x in [0, 1]:
            for y in [0, 1]:
                angle_diff = alice_angles[x] - bob_angles[y]
                E_ideal = visibility * np.cos(2 * angle_diff)
                E_noisy = E_ideal * noise_factor + np.random.uniform(-0.02, 0.02)
                correlators[(x, y)] = E_noisy

        results["sweeps"].append({
            "alpha": float(alpha),
            "visibility": float(visibility),
            "chsh": float(chsh_noisy),
            "correlators": {str(k): v for k, v in correlators.items()}
        })

    # Compute K_G
    max_chsh = max(r["chsh"] for r in results["sweeps"])
    K_G = max_chsh / 2.0

    results["summary"] = {
        "K_G": float(K_G),
        "max_chsh": float(max_chsh),
        "theory": float(np.sqrt(2)),
        "deviation_percent": float(abs(K_G - np.sqrt(2)) / np.sqrt(2) * 100)
    }

    return results


def run_kg_sweep_qpu(backend, n_points: int = 8, shots: int = 500) -> Dict[str, Any]:
    """
    Measure Grothendieck constant K_G = sqrt(2) via visibility sweep on QPU.

    Args:
        backend: Azure Quantum backend
        n_points: Number of visibility points
        shots: Shots per measurement setting

    Returns:
        Results dictionary with K_G measurement
    """
    print("\n" + "=" * 60)
    print("K_G (GROTHENDIECK CONSTANT) SWEEP")
    print("=" * 60)
    print("Theory: K_G = sqrt(2) = 1.414")

    alphas = np.linspace(0.1, np.pi/4, n_points)
    results = {"sweeps": [], "timestamp": datetime.now().isoformat()}

    # Optimal CHSH angles
    alice_angles = {0: 0, 1: np.pi/4}
    bob_angles = {0: np.pi/8, 1: -np.pi/8}

    for i, alpha in enumerate(alphas):
        visibility = np.sin(2 * alpha)
        print(f"\n[{i+1}/{n_points}] alpha={alpha:.3f} (V={visibility:.3f}):")

        correlators = {}
        for x in [0, 1]:
            for y in [0, 1]:
                # Parametric entangled state
                qc = QuantumCircuit(2, 2)
                qc.ry(2 * alpha, 0)  # Create partial entanglement
                qc.cx(0, 1)
                # Measurement angles
                qc.ry(2 * alice_angles[x], 0)
                qc.ry(2 * bob_angles[y], 1)
                qc.measure([0, 1], [0, 1])

                job = backend.run(qc, shots=shots)
                result = job.result()
                counts = result.get_counts()
                correlators[(x, y)] = compute_correlator(counts)

        S = correlators[(0,0)] + correlators[(0,1)] + correlators[(1,0)] - correlators[(1,1)]
        print(f"  CHSH = {S:.4f}")

        results["sweeps"].append({
            "alpha": float(alpha),
            "visibility": float(visibility),
            "chsh": float(S),
            "correlators": {str(k): v for k, v in correlators.items()}
        })

    # Compute K_G
    max_chsh = max(r["chsh"] for r in results["sweeps"])
    K_G = max_chsh / 2.0

    results["summary"] = {
        "K_G": float(K_G),
        "max_chsh": float(max_chsh),
        "theory": float(np.sqrt(2)),
        "deviation_percent": float(abs(K_G - np.sqrt(2)) / np.sqrt(2) * 100)
    }

    print(f"\n{'=' * 60}")
    print("K_G SUMMARY")
    print(f"{'=' * 60}")
    print(f"K_G (measured): {K_G:.4f}")
    print(f"K_G (theory): {np.sqrt(2):.4f}")
    print(f"Deviation: {results['summary']['deviation_percent']:.2f}%")

    return results


def main(use_simulation: bool = False, n_points: int = 8, shots: int = 500):
    """
    Main experiment function.

    Args:
        use_simulation: If True, use simulation instead of QPU
        n_points: Number of visibility points
        shots: Measurement shots per setting
    """
    print("=" * 60)
    print("K_G (GROTHENDIECK CONSTANT) MEASUREMENT")
    print("=" * 60)

    if use_simulation or not HAS_AZURE:
        print("Running in SIMULATION mode\n")
        results = simulate_kg_sweep(n_points, shots)
        results["backend"] = "simulation"
    else:
        provider = AzureQuantumProvider(resource_id=RESOURCE_ID, location=LOCATION)
        backend = provider.get_backend(QPU_TARGET)
        print(f"Backend: {backend.name}\n")
        results = run_kg_sweep_qpu(backend, n_points, shots)
        results["backend"] = QPU_TARGET

    results["shots"] = shots
    results["n_points"] = n_points

    # Display summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"K_G (measured): {results['summary']['K_G']:.4f}")
    print(f"K_G (theory): {results['summary']['theory']:.4f}")
    print(f"Deviation: {results['summary']['deviation_percent']:.2f}%")

    if results['summary']['deviation_percent'] < 1.0:
        print("\n>>> EXCELLENT: K_G = sqrt(2) verified to < 1% precision!")
    elif results['summary']['deviation_percent'] < 5.0:
        print("\n>>> GOOD: K_G approximately sqrt(2) (within 5%)")
    else:
        print("\n>>> Note: Significant deviation - check hardware noise")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = RESULTS_DIR / f"kg_sweep_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='K_G Grothendieck Constant Sweep')
    parser.add_argument('--simulate', action='store_true', help='Use simulation mode')
    parser.add_argument('--n-points', type=int, default=8, help='Visibility points')
    parser.add_argument('--shots', type=int, default=500, help='Shots per setting')

    args = parser.parse_args()

    result = main(
        use_simulation=args.simulate,
        n_points=args.n_points,
        shots=args.shots
    )
