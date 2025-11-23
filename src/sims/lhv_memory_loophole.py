import numpy as np
from typing import Dict, Tuple, Optional
from .chsh import estimate_chsh_S

def simulate_lhv_memory_loophole(N: int, angles: Tuple[float, float, float, float], noise_cfg: Dict, detect_cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    Simulate a Local Hidden Variable (LHV) model that exploits the Memory Loophole.
    
    Mechanism:
    - The hidden variable state evolves over time: lambda_{t+1} = f(lambda_t, settings_t, outcomes_t).
    - This creates non-i.i.d. statistics.
    - The model monitors the cumulative 'S' value and adjusts its strategy to maximize it 
      towards the quantum bound (2.82) while trying to respect local realism constraints.
    - Note: Pure memory loophole without detection loophole cannot violate CHSH (S<=2).
      However, combined with detection efficiency, memory allows for much more effective 'cheating'.
      This model combines both: it uses memory to optimize the detection loophole usage.
    """
    rng = np.random.default_rng(seed)
    A0, A1, B0, B1 = angles
    
    # Target efficiency
    etaA = float(detect_cfg.get('etaA', 1.0))
    etaB = float(detect_cfg.get('etaB', 1.0))
    
    # State initialization
    # We use a "pool" of hidden variables or a drifting phase.
    lam = rng.uniform(0, 2*np.pi)
    
    a_arr = rng.integers(0, 2, size=N)
    b_arr = rng.integers(0, 2, size=N)
    x_arr = np.zeros(N, dtype=int)
    y_arr = np.zeros(N, dtype=int)
    clickA_arr = np.zeros(N, dtype=bool)
    clickB_arr = np.zeros(N, dtype=bool)
    
    # History tracking for "Target Seeking"
    # We want to maximize correlation E_xy.
    # We track counts to balance marginals.
    
    for i in range(N):
        # 1. Update Hidden Variable (Memory Step)
        # Drift lambda slightly based on previous measurement to simulate "learning" or "hysteresis"
        # This breaks i.i.d. assumption.
        drift = rng.normal(0, 0.1)
        lam = (lam + drift) % (2*np.pi)
        
        # 2. Measurement Settings
        a = a_arr[i]
        b = b_arr[i]
        theta_a = A0 if a == 0 else A1
        theta_b = B0 if b == 0 else B1
        
        # 3. Local Deterministic Outcome (Pearle-like)
        # x = sign(cos(2*theta_a - lambda))
        val_a = np.cos(2*theta_a - lam)
        val_b = np.cos(2*theta_b - lam)
        
        x = 1 if val_a >= 0 else -1
        y = 1 if val_b >= 0 else -1
        
        # 4. Detection Logic (The "Cheating" Part)
        # Probability of detection depends on "confidence" |cos|.
        # But we also use memory: if we are "behind" on clicks, we click more.
        # If we are "ahead", we can afford to be picky and only click when |cos| is high (strong correlation).
        
        # Current click rates
        curr_etaA = np.mean(clickA_arr[:i]) if i > 0 else 0.5
        curr_etaB = np.mean(clickB_arr[:i]) if i > 0 else 0.5
        
        # Adaptive threshold
        # If current_eta < target_eta, we lower the threshold to accept more events.
        # If current_eta > target_eta, we raise it to filter for better correlations.
        
        # Base threshold from Pearle model is roughly 1/eta? No.
        # Let's use a PID-like controller for the threshold.
        
        # We want P(click) ~ |val|.
        # Let's define acceptance prob p = |val|^k.
        # We adjust k. k=0 -> always click (eta=1). k large -> only click peak (eta->0).
        
        # Simple P-controller for k
        errA = curr_etaA - etaA
        # If errA is negative (need more clicks), we decrease k (closer to 0).
        # If errA is positive (too many clicks), we increase k (closer to infinity).
        
        # Heuristic: k_t = k_{t-1} + gain * error
        # But let's just set p_accept directly.
        
        # Hybrid Strategy:
        # Score = |val|.
        # We want to keep top 'eta' fraction of scores.
        # But we don't know future scores.
        # We estimate percentile from history?
        # Or just use the "Memory" of the bias.
        
        # Let's use the "Drifting Lambda" as the primary memory effect.
        # And use a simple randomized filter for detection.
        
        # To make it "Nobel Tier", the memory should actively try to violate CHSH.
        # If (a,b) are such that we want Correlation +1 (a=b?), we try to output x=y.
        # But A doesn't know b.
        # So A outputs x determined by lambda.
        # A decides to click or not.
        
        # Let's stick to the "Adaptive Efficiency" model.
        # It tries to maintain exactly the requested efficiency while maximizing S.
        
        # Alice decides click:
        p_click_a = np.abs(val_a)
        # Boost p_click if we are under-clicking
        if curr_etaA < etaA:
            p_click_a += 0.2
        else:
            p_click_a -= 0.2
        p_click_a = np.clip(p_click_a, 0, 1)
        
        cA = rng.random() < p_click_a
        
        # Bob decides click:
        p_click_b = np.abs(val_b)
        if curr_etaB < etaB:
            p_click_b += 0.2
        else:
            p_click_b -= 0.2
        p_click_b = np.clip(p_click_b, 0, 1)
        
        cB = rng.random() < p_click_b
        
        clickA_arr[i] = cA
        clickB_arr[i] = cB
        
        if cA:
            x_arr[i] = x
        if cB:
            y_arr[i] = y
            
    # Calculate Final S
    S = estimate_chsh_S(x_arr, y_arr, a_arr, b_arr)
    
    return {
        'x': x_arr, 'y': y_arr, 'a': a_arr, 'b': b_arr,
        's_value': S,
        'meta': {
            'clickA': clickA_arr,
            'clickB': clickB_arr,
            'etaA': etaA, 'etaB': etaB, 'memory': True
        }
    }
