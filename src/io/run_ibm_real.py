import argparse
import numpy as np
import pandas as pd
import time
import sys
import os
from typing import List, Dict

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try importing qiskit
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError:
    print("Qiskit not installed. Please install: pip install qiskit qiskit-ibm-runtime qiskit-aer")
    exit(1)

def create_bell_circuit(theta_a: float, theta_b: float) -> QuantumCircuit:
    """
    Create a CHSH circuit.
    Alice measures in basis rotated by theta_a.
    Bob measures in basis rotated by theta_b.
    State: |Phi+> = (|00> + |11>)/sqrt(2)
    """
    qc = QuantumCircuit(2, 2)
    
    # Create Bell State |Phi+>
    qc.h(0)
    qc.cx(0, 1)
    
    # Alice's measurement basis (Qubit 0)
    # Rotate Y by theta_a
    qc.ry(theta_a, 0)
    
    # Bob's measurement basis (Qubit 1)
    # Rotate Y by theta_b
    qc.ry(theta_b, 1)
    
    # Measure
    qc.measure([0, 1], [0, 1])
    
    return qc

def run_ibm_experiment(api_token: str, backend_name: str = 'ibm_brisbane', shots: int = 1024):
    """
    Run the full CHSH experiment on IBM Quantum.
    """
    print(f"Connecting to IBM Quantum with token ending in ...{api_token[-4:]}")
    
    # 1. Authenticate
    service = None
    try:
        # Try ibm_cloud channel first
        service = QiskitRuntimeService(channel="ibm_cloud", token=api_token)
        print("Successfully authenticated with ibm_cloud channel")
    except Exception as e1:
        print(f"ibm_cloud authentication failed: {e1}")
        try:
            # Try ibm_quantum channel
            service = QiskitRuntimeService(token=api_token)
            print("Successfully authenticated with default channel")
        except Exception as e2:
            print(f"Default authentication also failed: {e2}")
            print("Falling back to local simulator...")
            backend_name = 'simulator'

    # 2. Select Backend
    if backend_name == 'simulator':
        backend = AerSimulator()
        print("Using local AerSimulator (Mock Real Hardware)")
    elif backend_name == 'least_busy':
        print("Finding least busy real backend...")
        try:
            # service is already defined in step 1? No, step 1 failed because of this line.
            # We need to ensure service is defined.
            # Actually, step 1 is BEFORE this block.
            # So we just use 'service' which was defined in step 1.
            
            # Filter for real backends (not simulators) that are operational
            backends = service.backends(simulator=False, operational=True)
            if not backends:
                print("No real backends available. Falling back to simulator.")
                backend = AerSimulator()
            else:
                # Find least busy
                # We need to check status. This might be slow.
                # Heuristic: just pick the first one or use a helper if available.
                # Qiskit's `least_busy` function is in `qiskit_ibm_provider` usually, but let's try manual sort if needed.
                # Actually, let's just pick 'ibm_brisbane' or similar if available, or the first one.
                # Better: let's try to get the one with the shortest queue.
                best_backend = None
                min_jobs = float('inf')
                
                for b in backends:
                    try:
                        stat = b.status()
                        jobs = stat.pending_jobs
                        if jobs < min_jobs:
                            min_jobs = jobs
                            best_backend = b
                    except:
                        continue
                
                if best_backend:
                    backend = best_backend
                    print(f"Selected least busy backend: {backend.name} (Pending jobs: {min_jobs})")
                else:
                    backend = backends[0]
                    print(f"Selected backend: {backend.name}")
        except Exception as e:
            print(f"Error finding backend: {e}")
            return
    else:
        try:
            backend = service.backend(backend_name)
            print(f"Using backend: {backend.name}")
        except:
            print(f"Backend {backend_name} not found. Listing available:")
            try:
                print(service.backends())
            except:
                print("Could not list backends.")
            return

    # 3. Define CHSH Angles
    # Standard CHSH angles (optimal violation)
    # A0 = 0, A1 = pi/2
    # B0 = pi/4, B1 = -pi/4 (or 3pi/4)
    # Note: Qiskit Ry(theta) rotates by theta/2? No, Ry(theta) is exp(-i theta Y / 2).
    # We need to check the convention.
    # Standard: A0=0, A1=pi/2. B0=pi/4, B1=-pi/4.
    # Ry(theta) rotates state. Measurement in basis M means rotate state by -M then measure Z.
    
    # Let's use standard settings:
    # A0 (Z-basis) -> theta=0
    # A1 (X-basis) -> theta=-pi/2 (Ry(-pi/2) maps X to Z)
    # B0 (Z+X)/sqrt(2) -> theta=-pi/4
    # B1 (Z-X)/sqrt(2) -> theta=pi/4
    
    angles = [
        (0, -np.pi/4, 0, 0), # A0, B0
        (0, np.pi/4, 0, 1),  # A0, B1
        (-np.pi/2, -np.pi/4, 1, 0), # A1, B0
        (-np.pi/2, np.pi/4, 1, 1)   # A1, B1
    ]
    
    circuits = []
    metadata = []
    
    for (thA, thB, a, b) in angles:
        qc = create_bell_circuit(thA, thB)
        circuits.append(qc)
        metadata.append({'a': a, 'b': b})
        
    # 4. Transpile
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuits = pm.run(circuits)
    
    # 5. Run
    print(f"Submitting job to {backend.name}...")
    sampler = Sampler(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    print(f"Job ID: {job.job_id()}")
    
    result = job.result()
    
    # 6. Parse Results
    rows = []
    for i, pub_result in enumerate(result):
        # pub_result is a PubResult
        # Get counts
        data = pub_result.data.c.get_counts()
        # data is {'00': count, '01': count...}
        
        a = metadata[i]['a']
        b = metadata[i]['b']
        
        for bitstring, count in data.items():
            # Qiskit bitstring is Little Endian? "bit1 bit0" -> "Bob Alice"
            # Let's assume standard: q0 is rightmost? No, qiskit is q_n ... q_0.
            # So '01' means q1=0, q0=1.
            # We mapped q0->Alice, q1->Bob.
            # So '01' -> Bob=0, Alice=1.
            
            outcome_alice = int(bitstring[1]) # q0
            outcome_bob = int(bitstring[0])   # q1
            
            # Map 0/1 to +1/-1
            x = 1 if outcome_alice == 0 else -1
            y = 1 if outcome_bob == 0 else -1
            
            # Add 'count' rows
            for _ in range(count):
                rows.append({'a': a, 'b': b, 'x': x, 'y': y})
                
    df = pd.DataFrame(rows)
    
    # Calculate S
    from src.sims.chsh import estimate_chsh_S
    S = estimate_chsh_S(df['x'].values, df['y'].values, df['a'].values, df['b'].values)
    print(f"Experiment Complete. S-value = {S:.4f}")
    
    # Save
    timestamp = int(time.time())
    filename = f"ibm_data_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bell Test on IBM Quantum")
    parser.add_argument("--token", type=str, required=True, help="IBM Quantum API Token")
    parser.add_argument("--backend", type=str, default="simulator", help="Backend name (e.g. ibm_brisbane)")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per circuit")
    
    args = parser.parse_args()
    run_ibm_experiment(args.token, args.backend, args.shots)
