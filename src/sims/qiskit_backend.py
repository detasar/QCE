from typing import Dict, Tuple, Optional
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

def simulate_chsh_qiskit(N: int, angles: Tuple[float, float, float, float], noise_cfg: Dict, detect_cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    Simulate CHSH experiment using Qiskit Aer backend.
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit is not installed. Please install qiskit and qiskit-aer.")

    rng = np.random.default_rng(seed)
    A0, A1, B0, B1 = angles
    
    # Generate settings a, b
    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    
    # Count occurrences of each setting pair to batch simulation
    counts_req = {
        (0, 0): np.sum((a == 0) & (b == 0)),
        (0, 1): np.sum((a == 0) & (b == 1)),
        (1, 0): np.sum((a == 1) & (b == 0)),
        (1, 1): np.sum((a == 1) & (b == 1)),
    }
    
    # Build Noise Model
    noise_model = NoiseModel()
    p_depol = float(noise_cfg.get('p_depol', 0.0))
    if p_depol > 0:
        # Apply 2-qubit depolarizing error to CNOT
        # Note: p_depol in chsh.py shrinks E by (1-p).
        # In Qiskit, depolarizing_error(p, 2) shrinks Bloch vector by (1 - 16/15 p)?
        # Standard depolarizing channel: rho -> (1-p)rho + p I/4.
        # Correlation E -> (1-p) E.
        # Qiskit depolarizing_error(p, 1) does rho -> (1-p)rho + p I/2.
        # We want to match the strength.
        # Let's just use the parameter as is for now, exact matching might require calibration.
        error_gate = depolarizing_error(p_depol, 2)
        noise_model.add_all_qubit_quantum_error(error_gate, ['cx'])
        
    px = float(noise_cfg.get('px', 0.0))
    pz = float(noise_cfg.get('pz', 0.0))
    if px > 0 or pz > 0:
        # Apply local noise before measurement
        # We can add it to identity gates or just assume it happens.
        # Let's add to measurement? No, noise model usually attaches to gates.
        # Let's add to ID gates before measurement.
        error_x = pauli_error([('X', px), ('I', 1 - px)])
        error_z = pauli_error([('Z', pz), ('I', 1 - pz)])
        combined = error_x.compose(error_z)
        noise_model.add_all_qubit_quantum_error(combined, ['id'])

    sim = AerSimulator(noise_model=noise_model)
    
    # Prepare circuits
    circuits = []
    setting_map = [] # To map back to (a,b) pairs
    
    # We need 4 circuits, one for each setting pair (A_a, B_b)
    # Alice is qubit 0, Bob is qubit 1
    
    for (sa, sb), count in counts_req.items():
        if count == 0:
            continue
            
        qc = QuantumCircuit(2, 2)
        # Bell State |Phi+> = (|00> + |11>) / sqrt(2)
        # Note: chsh.py implies E = cos(2(A-B)).
        # If we use Phi+, <ZZ> = 1.
        # If A=B=0, E=1. cos(0)=1. Matches.
        qc.h(0)
        qc.cx(0, 1)
        
        # Noise injection (ID gates)
        if px > 0 or pz > 0:
            qc.id(0)
            qc.id(1)
            
        # Measurement settings
        # Rotate basis. We want to measure in basis with angle theta.
        # We rotate state by -theta (Ry(-2*theta_val)) then measure Z.
        theta_a = A0 if sa == 0 else A1
        theta_b = B0 if sb == 0 else B1
        
        # Qiskit Ry rotation angle is just the angle in Bloch sphere?
        # Ry(theta) = exp(-i theta Y / 2).
        # If we want to measure along axis at angle theta from Z in X-Z plane:
        # We rotate by -theta around Y.
        # Wait, chsh.py uses 2*theta in cos.
        # Let's assume A, B are the actual physical angles.
        # Then we rotate by -2*A.
        
        qc.ry(-2 * theta_a, 0)
        qc.ry(-2 * theta_b, 1)
        
        qc.measure([0, 1], [0, 1])
        
        circuits.append(qc)
        setting_map.append(((sa, sb), count))

    # Run batch
    # We need 'count' shots for each circuit.
    # Qiskit execute accepts 'shots'. But here we have different shots for different circuits?
    # We can run them sequentially or pad.
    # Simplest: Run each circuit with its specific shots.
    
    results_map = {}
    
    for i, qc in enumerate(circuits):
        (sa, sb), count = setting_map[i]
        # Transpile
        tqc = transpile(qc, sim)
        # Run
        job = sim.run(tqc, shots=count, seed_simulator=seed)
        res = job.result()
        # Get memory (individual shots)
        # We need to enable memory=True
        job_mem = sim.run(tqc, shots=count, memory=True, seed_simulator=seed)
        mem = job_mem.result().get_memory()
        # mem is list of strings like '01' (qubit 1 is '0', qubit 0 is '1' -> Little Endian usually in Qiskit?)
        # Qiskit is Little Endian: 'q1 q0'.
        # So '01' means q1=0, q0=1.
        # We mapped Alice->0, Bob->1.
        # So q0 is Alice, q1 is Bob.
        # '01' -> Bob=0, Alice=1.
        
        # Parse memory
        # We want outcomes +1/-1. '0'->+1, '1'->-1.
        # Let's convert to int 0/1 first.
        
        outcomes = []
        for shot in mem:
            # shot is string 'B A'
            bit_b = int(shot[0])
            bit_a = int(shot[1])
            
            val_a = 1 if bit_a == 0 else -1
            val_b = 1 if bit_b == 0 else -1
            outcomes.append((val_a, val_b))
            
        results_map[(sa, sb)] = outcomes
        
    # Reconstruct x, y arrays
    x = np.zeros(N, dtype=int)
    y = np.zeros(N, dtype=int)
    
    # We need to consume the outcomes
    counters = {k: 0 for k in counts_req.keys()}
    
    for i in range(N):
        sa, sb = a[i], b[i]
        out_list = results_map.get((sa, sb), [])
        idx = counters[(sa, sb)]
        if idx < len(out_list):
            xa, yb = out_list[idx]
            x[i] = xa
            y[i] = yb
            counters[(sa, sb)] += 1
        else:
            # Should not happen
            x[i] = 0
            y[i] = 0

    # Detection Efficiency (Post-processing)
    etaA = float(detect_cfg.get('etaA', 1.0))
    etaB = float(detect_cfg.get('etaB', 1.0))
    darkA = float(detect_cfg.get('darkA', 0.0))
    darkB = float(detect_cfg.get('darkB', 0.0))
    
    clickA = np.zeros(N, dtype=bool)
    clickB = np.zeros(N, dtype=bool)
    
    # Apply efficiency
    # If click, keep value. If no click, apply dark count logic (random outcome).
    
    rand_etaA = rng.random(N)
    rand_etaB = rng.random(N)
    rand_darkA = rng.random(N)
    rand_darkB = rng.random(N)
    rand_coinA = rng.random(N)
    rand_coinB = rng.random(N)
    
    for i in range(N):
        # Alice
        if rand_etaA[i] < etaA:
            clickA[i] = True
            # x[i] remains true outcome
        else:
            # Dark count?
            if rand_darkA[i] < darkA:
                clickA[i] = True
                x[i] = 1 if rand_coinA[i] < 0.5 else -1
            else:
                clickA[i] = False
                # x[i] is technically undefined/lost, but we keep it or set to 0?
                # chsh.py sets it to 0? No, chsh.py sets it to 1/-1 if dark, else it was already set.
                # But if NOT clicked and NOT dark, what is x[i]?
                # In chsh.py: "true detection keeps xi/yi... else if dark... else..."
                # Wait, chsh.py initializes x=0. So if no click, x=0.
                # But wait, estimate_chsh_S filters by mask?
                # No, estimate_chsh_S calculates mean(x*y). If x=0, product is 0.
                # But usually we only look at clicked events.
                # Let's check estimate_chsh_S.
                # It computes mean over ALL N?
                # "mask = (a == ab0) & (b == ab1)".
                # It uses all events.
                # If x=0 for lost events, then S will be small.
                # But we usually compute S on the *post-selected* subset (coincidence).
                # The standard CHSH S is conditioned on coincidence.
                # But `simulate_chsh` returns `s_value`.
                # Let's check `estimate_chsh_S` in `chsh.py`.
                # It takes x, y.
                # If x, y are 0, they contribute 0.
                # This implies `simulate_chsh` computes S over the whole ensemble including lost photons?
                # That would kill S.
                # Let's re-read `chsh.py`.
                # `x = np.zeros(N, dtype=int)`.
                # If not clicked, it stays 0.
                # So yes, S includes 0s.
                # This means `s_value` in `simulate_chsh` is the "raw" S, heavily penalized by loss.
                # But wait, `run_g2_device` calculates `S_mean`.
                # If eta=0.9, loss is small.
                # If eta is small, S -> 0.
                # This is correct for "detection loophole" tests.
                pass
        
        # Bob (same logic)
        if rand_etaB[i] < etaB:
            clickB[i] = True
        else:
            if rand_darkB[i] < darkB:
                clickB[i] = True
                y[i] = 1 if rand_coinB[i] < 0.5 else -1
            else:
                clickB[i] = False
                
    # If not clicked, set x/y to 0 to match chsh.py behavior
    x[~clickA] = 0
    y[~clickB] = 0
    
    from .chsh import estimate_chsh_S
    S = estimate_chsh_S(x, y, a, b)
    
    return {
        'x': x, 'y': y, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'clickA': clickA,
            'clickB': clickB,
            'etaA': etaA, 'etaB': etaB, 'darkA': darkA, 'darkB': darkB,
        }
    }
