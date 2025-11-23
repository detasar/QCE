import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Tuple

def contextual_fraction(p_dist: Dict[Tuple[int, int], Dict[Tuple[int, int], float]]) -> float:
    """
    Calculate the Contextual Fraction (CF) of an empirical behavior using Linear Programming.
    Based on Abramsky & Brandenburger (2011).
    
    The behavior P(a,b|x,y) is contextual if it cannot be decomposed into a convex mixture
    of deterministic non-contextual (local) models.
    
    CF(P) = 1 - max(lambda) such that P = lambda * P_NC + (1-lambda) * P_C
    where P_NC is a non-contextual behavior.
    
    This is equivalent to finding the maximum weight of a local model that fits 'under' P.
    
    Args:
        p_dist: Dictionary mapping settings (a,b) -> {outcomes (x,y) -> prob}
    
    Returns:
        CF: A float between 0 and 1. 0 = Non-contextual (Local), 1 = Strongly Contextual.
    """
    
    # 1. Enumerate all deterministic local strategies (vertices of the local polytope)
    # For CHSH: A inputs {0,1}, B inputs {0,1}. Outputs {0,1} (or +/-1).
    # A strategy is defined by functions fA: {0,1}->{0,1} and fB: {0,1}->{0,1}.
    # There are 4 functions for A (00, 01, 10, 11) and 4 for B.
    # Total 16 deterministic local strategies.
    
    strategies = []
    for fa_code in range(4): # 2 bits for fA(0), fA(1)
        for fb_code in range(4): # 2 bits for fB(0), fB(1)
            # Decode functions
            fA = {0: (fa_code >> 1) & 1, 1: fa_code & 1}
            fB = {0: (fb_code >> 1) & 1, 1: fb_code & 1}
            
            # Build probability vector for this strategy
            # Vector format: P(00|00), P(01|00), P(10|00), P(11|00), P(00|01)...
            # 4 settings * 4 outcomes = 16 components
            strat_vec = []
            for a_set in [0, 1]:
                for b_set in [0, 1]:
                    ax = fA[a_set]
                    by = fB[b_set]
                    
                    # Outcomes (0,0), (0,1), (1,0), (1,1)
                    for out_x in [0, 1]:
                        for out_y in [0, 1]:
                            if out_x == ax and out_y == by:
                                strat_vec.append(1.0)
                            else:
                                strat_vec.append(0.0)
            strategies.append(strat_vec)
            
    A_eq = np.array(strategies).T # Columns are strategies
    # A_eq shape: (16, 16)
    
    # 2. Construct the empirical probability vector P_emp
    # We need to map the input dictionary to the same vector format.
    P_emp = []
    # Normalize input just in case
    
    # Helper to get prob
    def get_p(a, b, x, y):
        # Try to find in dict. Outcomes might be +/-1 or 0/1.
        # We assume 0/1 here for indexing.
        # If input is +/-1, map -1->0, 1->1? Or -1->0, 1->1.
        # Let's assume input dictionary uses consistent keys.
        d = p_dist.get((a,b), {})
        return d.get((x,y), 0.0)

    for a_set in [0, 1]:
        for b_set in [0, 1]:
            for out_x in [0, 1]:
                for out_y in [0, 1]:
                    P_emp.append(get_p(a_set, b_set, out_x, out_y))
                    
    P_emp = np.array(P_emp)
    
    # 3. Linear Program
    # We want to maximize sum(w_i) such that A_eq * w <= P_emp
    # Wait, the definition is P = sum(w_i * P_i) + P_noise?
    # No, CF is defined via the "Non-Contextual Fraction" NCF.
    # NCF = max sum(w_i) s.t. sum(w_i P_i) <= P_emp (component-wise)
    # This means the local part "fits inside" the empirical distribution.
    # The remainder is the contextual part.
    
    # Variables: w_0 ... w_15 (weights of local strategies)
    # Objective: Maximize sum(w) -> Minimize -sum(w)
    c = -1.0 * np.ones(16)
    
    # Constraints: A_eq * w <= P_emp
    # Bounds: w >= 0
    
    res = linprog(c, A_ub=A_eq, b_ub=P_emp, bounds=(0, None), method='highs')
    
    if res.success:
        NCF = -res.fun # Max weight
        CF = 1.0 - NCF
        return float(CF)
    else:
        return float('nan')

def sheaf_calibrated_threshold(alpha: float, p_dist: Dict, base_L: float, gamma: float = 1.0) -> float:
    """
    Adjust the CEW threshold based on the Contextual Fraction.
    Hypothesis: If CF is high, we are in a quantum regime, so we can lower the threshold
    to be more sensitive (since classical noise is less likely to mimic high CF).
    
    L_sheaf = base_L * (1 - gamma * CF)
    """
    cf = contextual_fraction(p_dist)
    if np.isnan(cf):
        return base_L
        
    # If CF is high (e.g. 0.4), we reduce L.
    # If CF is 0 (classical), we keep L (or increase it?).
    # Actually, if CF is 0, we need the strict bound.
    # If CF > 0, we have evidence of quantumness, so maybe we don't need the witness?
    # No, the witness is for *finite statistics* certification.
    # But this is a "Contextual-Aware" witness.
    
    return base_L * (1.0 - gamma * cf)
