import numpy as np
from typing import Dict, Tuple, Optional

def simulate_lhv_communication(N: int, angles: Tuple[float, float, float, float], 
                                noise_cfg: Dict, detect_cfg: Dict, 
                                seed: Optional[int] = None) -> Dict:
    """
    Simulate a "Communication Loophole" LHV model.
    
    This is a CLASSICAL model that violates locality by allowing Alice to send
    1 bit of information to Bob AFTER her measurement but BEFORE Bob's.
    
    Strategy:
    - Alice measures first, gets outcome x ∈ {-1, +1}
    - Alice sends signal: "I got x" to Bob
    - Bob uses this info to choose his outcome strategically
    - This can achieve S ≈ 2.5 (better than local but worse than quantum)
    
    Key: This is NOT quantum. It's classical signaling.
    The ML model should learn to distinguish this from genuine entanglement.
    """
    rng = np.random.default_rng(seed)
    A0, A1, B0, B1 = angles
    
    # Measurement settings
    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    
    # Hidden variable (shared randomness, but not sufficient alone)
    lam = rng.uniform(0, 2*np.pi, size=N)
    
    # Alice measures first (deterministically from lambda)
    theta_a = np.where(a == 0, A0, A1)
    x_raw = np.sign(np.cos(theta_a - lam))
    x_raw[x_raw == 0] = 1
    x = x_raw.astype(int)
    
    # Bob receives Alice's outcome x (this is the "cheating" part)
    # Bob's strategy: maximize correlation E[xy | a, b]
    # Optimal: if expecting positive correlation (a=0,b=0 or a=1,b=0), output y=x
    #          if expecting negative (a=0,b=1 or a=1,b=1), output y=-x
    
    # CHSH correlation pattern:
    # E(A0 B0) > 0, E(A0 B1) > 0, E(A1 B0) > 0, E(A1 B1) < 0
    
    # Bob's deterministic strategy based on signal
    y = np.zeros(N, dtype=int)
    
    for i in range(N):
        if (a[i] == 0 and b[i] == 0) or (a[i] == 1 and b[i] == 0) or (a[i] == 0 and b[i] == 1):
            # Expect positive correlation
            y[i] = x[i]
        else:
            # (a=1, b=1): expect negative
            y[i] = -x[i]
    
    # Add some noise to make it realistic
    noise_rate = 0.05
    flip = rng.random(N) < noise_rate
    y[flip] = -y[flip]
    
    # Detection
    etaA = float(detect_cfg.get('etaA', 1.0))
    etaB = float(detect_cfg.get('etaB', 1.0))
    
    clickA = rng.random(N) < etaA
    clickB = rng.random(N) < etaB
    
    x_out = np.where(clickA, x, 0)
    y_out = np.where(clickB, y, 0)
    
    from src.sims.chsh import estimate_chsh_S
    S = estimate_chsh_S(x_out, y_out, a, b)
    
    return {
        'x': x_out, 'y': y_out, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'source': 'communication_loophole',
            'clickA': clickA,
            'clickB': clickB,
            'etaA': etaA, 'etaB': etaB
        }
    }
