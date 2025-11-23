import numpy as np
from typing import Dict, Tuple, Optional

def simulate_lhv_garg_mermin(N: int, angles: Tuple[float, float, float, float], noise_cfg: Dict, detect_cfg: Dict, seed: Optional[int] = None) -> Dict:
    """
    Simulate a Local Hidden Variable (LHV) model that exploits the Detection Loophole (Garg-Mermin).
    
    Mechanism:
    - Hidden variable lambda is uniform on [0, 2pi).
    - Alice and Bob have local response functions A(a, lambda) and B(b, lambda).
    - Crucially, they can output 0 (no click) if the hidden variable is "unfavorable".
    - By selectively dropping events where correlations would be weak or wrong, 
      the remaining (coincident) events can violate CHSH inequalities.
    
    This model mimics the quantum cosine correlation E ~ cos(2(a-b)) perfectly 
    if efficiency is low enough (eta < 82.8%).
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0, 2*np.pi, size=N)
    A0, A1, B0, B1 = angles
    
    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    
    theta_a = np.where(a == 0, A0, A1)
    theta_b = np.where(b == 0, B0, B1)
    
    # Garg-Mermin / Larsson Model logic:
    # We want to mimic Quantum probability P(++|a,b) = 1/4 * (1 + cos(2(a-b)))
    # Standard LHV cannot do this.
    # But with post-selection:
    # Let local probability of detection be P(click|a, lambda).
    # If we set P(click) such that we filter out "bad" lambdas, we can shape the correlation.
    
    # Simple "Cosine Model" (Pearle 1970, Braunstein-Caves 1990):
    # Deterministic outcome v(a, lambda) = sign(cos(a - lambda))
    # Detection probability p(a, lambda) = |cos(a - lambda)|
    # This reproduces QM correlations exactly but requires eta <= 2/pi (~63.6%).
    # If eta is higher, we must click more often, diluting the correlation.
    
    # Implementation of Pearle's model:
    # Outcome is deterministic based on sign.
    # Detection is probabilistic based on magnitude.
    
    # Alice
    # We use 2*theta because our angles are half-angles (Bloch sphere).
    # The projection is onto angle 2*theta.
    # cos(2*theta - lambda)
    
    val_a_raw = np.cos(2*theta_a - lam)
    val_b_raw = np.cos(2*theta_b - lam) # Bob shares same lambda (anti-correlated? or same?)
    # For singlet, we usually have E = -cos(a-b).
    # Let's assume we want to mimic E = cos(2(a-b)) as in our chsh.py
    
    # Outcome is sign of projection
    x = np.sign(val_a_raw).astype(int)
    y = np.sign(val_b_raw).astype(int)
    x[x==0] = 1
    y[y==0] = 1 # bias?
    
    # Detection Probability
    # We want to simulate "efficiency".
    # If the requested eta is HIGH, this model fails (cannot hide enough).
    # If requested eta is LOW, this model works too well.
    # We will implement the "optimal" LHV strategy for a given eta.
    # But for simplicity, let's use the Pearle model and CLAMP the detection prob.
    
    prob_click_a = np.abs(val_a_raw)
    prob_click_b = np.abs(val_b_raw)
    
    # If real efficiency eta > 2/pi, we must click more.
    # We can mix this "cheating" strategy with a "honest" strategy.
    # But let's just use the cheating strategy and see if it passes as "Quantum".
    
    # We need to respect the user's 'eta' parameter.
    # If user sets eta=0.9, we MUST click 90% of the time.
    # The Pearle model only clicks 63% of the time.
    # So we must click on additional events.
    # Strategy: Click if |cos| > threshold? No, |cos| average is 2/pi.
    # To get higher efficiency, we must accept lower |cos| values.
    # This introduces errors (linear dependence instead of cosine).
    
    target_eta = float(detect_cfg.get('etaA', 0.9)) # Assume symmetric for now
    
    # Threshold logic: click if |cos| > T.
    # If T=0, eta=1. If T=1, eta=0.
    # For a given target_eta, we find T.
    # For uniform lambda, |cos| is uniform? No.
    # If lambda ~ U, y=cos(lambda). P(y) ~ 1/sqrt(1-y^2).
    # Actually, simpler:
    # We just use the probabilistic acceptance: P(click) = |cos|^n ?
    # Or P(click) = eta_eff + (1-eta_eff)*|cos|.
    
    # Let's use a hybrid model:
    # With prob 'p_cheat', we use the Pearle filter (|cos|).
    # With prob '1-p_cheat', we always click (or click with prob eta).
    # This mixes Quantum-like correlations with Classical linear ones.
    
    # Actually, let's just implement the "Standard" LHV (Garg-Mermin) which is:
    # x = sign(cos(2a - lambda))
    # y = sign(cos(2b - lambda))
    # This gives linear correlation (sawtooth).
    # This is what we want to test the ML against!
    # The previous "simulate_lhv" in mlw.py was essentially this but with random noise?
    # No, mlw.py's simulate_lhv was: x = sign(cos(a-lambda)).
    # That IS the standard LHV.
    # Wait, standard LHV gives E = 1 - 2|a-b|/pi (linear).
    # Quantum gives E = cos(2(a-b)).
    # The difference is subtle (curved vs straight).
    # ML detected this easily (100%).
    
    # To make it HARDER (Nobel Tier), we need the LHV to produce CURVED correlations.
    # This requires the Detection Loophole.
    # We will enforce the Pearle model logic:
    # Click probability depends on |cos|.
    # If the resulting click rate is lower than 'eta', we add random clicks to match 'eta'.
    
    clickA = np.zeros(N, dtype=bool)
    clickB = np.zeros(N, dtype=bool)
    
    # Base Pearle clicks
    # P(click) = |cos(2theta - lambda)|
    # This naturally gives ~63% efficiency.
    pearle_click_a = rng.random(N) < np.abs(val_a_raw)
    pearle_click_b = rng.random(N) < np.abs(val_b_raw)
    
    # If target_eta > 0.63, we need more clicks.
    # If target_eta < 0.63, we drop some.
    
    # Let's assume we are in the "Hard" regime where we try to fake QM as much as possible
    # given the efficiency constraint.
    
    # For this implementation, let's just use the Pearle clicks directly 
    # but scale them to match target_eta roughly.
    # P(click) = min(1, |cos| / threshold) ?
    
    # Let's try: P(click) = |cos|^k.
    # Tuning k allows changing the curve shape.
    # But to fake QM, k=1 is best (Pearle).
    
    # We will use the Pearle outcomes for x,y.
    # And we will use the Pearle detection logic.
    # BUT we will force the click rate to match target_eta by random filling if needed.
    
    current_eta = np.mean(pearle_click_a) # approx 0.63
    
    if target_eta > current_eta:
        # We need to add clicks.
        # The added clicks will be "honest" (uncorrelated/linear).
        # This dilutes the fake-QM signal.
        # This is exactly what physics predicts: at high eta, you can't fake QM.
        missing = target_eta - current_eta
        # Add random clicks to non-clicked events
        # We need to add 'missing' fraction of TOTAL events.
        # Available non-clicked is (1-current_eta).
        # Prob to add = missing / (1-current_eta).
        
        prob_add = missing / (1.0 - current_eta)
        prob_add = np.clip(prob_add, 0, 1)
        
        extra_a = (rng.random(N) < prob_add) & (~pearle_click_a)
        extra_b = (rng.random(N) < prob_add) & (~pearle_click_b)
        
        clickA = pearle_click_a | extra_a
        clickB = pearle_click_b | extra_b
        
    else:
        # We have too many clicks (target_eta < 0.63).
        # We can drop more to be even more Quantum-like?
        # Actually Pearle is optimal for eta=0.63.
        # If eta is lower, we can just subsample.
        prob_keep = target_eta / current_eta
        keep_a = rng.random(N) < prob_keep
        keep_b = rng.random(N) < prob_keep
        
        clickA = pearle_click_a & keep_a
        clickB = pearle_click_b & keep_b

    # Apply to x, y (if no click, value is 0 or hidden)
    # Our format uses 0 for no click in x,y arrays?
    # chsh.py uses 0.
    
    x_out = np.where(clickA, x, 0)
    y_out = np.where(clickB, y, 0)
    
    from .chsh import estimate_chsh_S
    S = estimate_chsh_S(x_out, y_out, a, b)
    
    return {
        'x': x_out, 'y': y_out, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'clickA': clickA,
            'clickB': clickB,
            'etaA': target_eta, 'etaB': target_eta, 'darkA': 0.0, 'darkB': 0.0,
        }
    }
