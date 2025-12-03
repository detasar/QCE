#!/usr/bin/env python3
"""
================================================================================
IONQ FORTE ENTERPRISE CHSH MEASUREMENT
================================================================================

This experiment runs CHSH Bell test on IonQ Forte Enterprise trapped-ion QPU.

PURPOSE:
--------
Validate Bell inequality violation on real trapped-ion quantum hardware.

METHOD:
-------
1. Prepare Bell state |Phi+> = (|00> + |11>)/sqrt(2)
2. Measure in four CHSH-optimal angle combinations
3. Compute CHSH S-value and Bell fidelity

HARDWARE:
---------
IonQ Forte Enterprise (Azure Quantum)
- 36 algorithmic qubits (171Yb+ trapped ions)
- All-to-all connectivity
- T2 ~ 1 second coherence
- Gate fidelity > 99%

EXPECTED RESULTS:
-----------------
CHSH S = 2.716 (96% of Tsirelson bound)
Bell Fidelity = 0.984 (near-ideal)

Author: Davut Emre Tasar
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Note: Requires Azure Quantum access for real QPU execution
try:
    from qiskit import QuantumCircuit
    from azure.quantum.qiskit import AzureQuantumProvider
    HAS_AZURE = True
except ImportError:
    HAS_AZURE = False
    print("Warning: Azure Quantum SDK not available. Using simulation mode.")


# Azure Quantum configuration (replace with your credentials)
RESOURCE_ID = '/subscriptions/YOUR_SUBSCRIPTION/resourceGroups/YOUR_GROUP/providers/Microsoft.Quantum/workspaces/YOUR_WORKSPACE'
LOCATION = 'westeurope'
QPU_TARGET = 'ionq.qpu.forte-enterprise-1'

OUTPUT_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def simulate_chsh(shots: int = 500) -> dict:
    """
    Simulate CHSH measurement for testing without hardware access.

    Args:
        shots: Number of measurement shots

    Returns:
        Simulated result dict
    """
    np.random.seed(42)

    # Simulate near-ideal Bell state measurements
    visibility = 0.96  # Typical IonQ visibility

    alice_angles = {0: 0, 1: np.pi/4}
    bob_angles = {0: np.pi/8, 1: -np.pi/8}

    correlators = {}
    all_counts = {}

    for x in [0, 1]:
        for y in [0, 1]:
            angle_diff = alice_angles[x] - bob_angles[y]
            p_same = (1 + visibility * np.cos(2 * angle_diff)) / 2

            # Generate counts
            n_same = int(shots * p_same)
            n_diff = shots - n_same

            counts = {
                '00': n_same // 2 + np.random.randint(-5, 5),
                '11': n_same // 2 + np.random.randint(-5, 5),
                '01': n_diff // 2 + np.random.randint(-5, 5),
                '10': n_diff // 2 + np.random.randint(-5, 5)
            }

            total = sum(counts.values())
            E = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / total
            correlators[(x, y)] = E
            all_counts[f"({x},{y})"] = counts

    S = correlators[(0,0)] + correlators[(0,1)] + correlators[(1,0)] - correlators[(1,1)]
    fidelity = (0.48 + 0.48 + np.random.uniform(-0.02, 0.02))  # ~0.96 typical

    return {
        "backend": "simulation",
        "timestamp": datetime.now().isoformat(),
        "shots": shots,
        "bell_fidelity": fidelity,
        "chsh_value": S,
        "correlators": {str(k): v for k, v in correlators.items()},
        "all_counts": all_counts,
        "classical_bound": 2.0,
        "tsirelson_bound": 2.828,
        "violation": S > 2.0
    }


def run_qpu_chsh(resource_id: str, shots: int = 500):
    """
    Run CHSH experiment on IonQ Forte Enterprise QPU.

    Args:
        resource_id: Azure Quantum workspace resource ID
        shots: Number of measurement shots per setting

    Returns:
        Result dictionary with CHSH value and statistics
    """
    print("=" * 60)
    print("REAL QPU: IonQ Forte Enterprise")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    provider = AzureQuantumProvider(resource_id=resource_id, location=LOCATION)
    backend = provider.get_backend(QPU_TARGET)
    print(f"Backend: {backend.name}")
    print("Note: QPU jobs may take 1-5 minutes in queue\n")

    # 1. Bell state fidelity
    print("=" * 40)
    print("1. BELL STATE FIDELITY")
    print("=" * 40)

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    print("Submitting Bell state job...")
    job = backend.run(qc, shots=shots)
    print(f"Job ID: {job.job_id()}")
    print("Waiting for result...")

    result = job.result()
    counts = result.get_counts()
    fidelity = (counts.get('00', 0) + counts.get('11', 0)) / shots

    print(f"Counts: {counts}")
    print(f"Fidelity: {fidelity:.4f}")

    # 2. CHSH measurement
    print("\n" + "=" * 40)
    print("2. CHSH MEASUREMENT")
    print("=" * 40)

    # CHSH optimal angles
    alice_angles = {0: 0, 1: np.pi/4}
    bob_angles = {0: np.pi/8, 1: -np.pi/8}

    correlators = {}
    all_counts = {}

    for x in [0, 1]:
        for y in [0, 1]:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.ry(2 * alice_angles[x], 0)
            qc.ry(2 * bob_angles[y], 1)
            qc.measure([0, 1], [0, 1])

            print(f"Running E({x},{y})...", end=" ", flush=True)
            job = backend.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()

            total = sum(counts.values())
            E = (counts.get('00', 0) - counts.get('01', 0) -
                 counts.get('10', 0) + counts.get('11', 0)) / total
            correlators[(x, y)] = E
            all_counts[f"({x},{y})"] = counts
            print(f"E = {E:.3f}")

    S = correlators[(0,0)] + correlators[(0,1)] + correlators[(1,0)] - correlators[(1,1)]

    print(f"\n{'=' * 40}")
    print("RESULTS")
    print("=" * 40)
    print(f"CHSH Value: {S:.4f}")
    print(f"Classical bound: 2.0")
    print(f"Tsirelson bound: 2.828")
    print(f"Bell Violation: {'YES' if S > 2.0 else 'NO'}")
    print(f"Bell Fidelity: {fidelity:.4f}")

    # Save results
    result_data = {
        "backend": QPU_TARGET,
        "timestamp": datetime.now().isoformat(),
        "shots": shots,
        "bell_fidelity": float(fidelity),
        "bell_counts": counts,
        "chsh_value": float(S),
        "correlators": {str(k): v for k, v in correlators.items()},
        "all_counts": all_counts,
        "classical_bound": 2.0,
        "tsirelson_bound": 2.828,
        "violation": S > 2.0
    }

    return result_data


def main(use_simulation: bool = False, shots: int = 500):
    """
    Main experiment function.

    Args:
        use_simulation: If True, use simulation instead of real QPU
        shots: Number of measurement shots
    """
    if use_simulation or not HAS_AZURE:
        print("Running in SIMULATION mode")
        result_data = simulate_chsh(shots)
    else:
        result_data = run_qpu_chsh(RESOURCE_ID, shots)

    # Save results
    output_file = OUTPUT_DIR / f"ionq_chsh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

    return result_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='IonQ CHSH Measurement')
    parser.add_argument('--simulate', action='store_true', help='Use simulation mode')
    parser.add_argument('--shots', type=int, default=500, help='Shots per setting')

    args = parser.parse_args()

    result = main(use_simulation=args.simulate, shots=args.shots)
