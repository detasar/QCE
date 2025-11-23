import pandas as pd
import numpy as np
from typing import Dict, Optional

def load_big_bell_test_data(filepath: str) -> Dict:
    """
    Load data from the Big Bell Test (or similar experiments).
    Expected CSV format:
    - setting_a (0/1)
    - setting_b (0/1)
    - outcome_a (0/1)
    - outcome_b (0/1)
    - (optional) timestamp
    
    Returns a dictionary compatible with our simulation format:
    {'x': ..., 'y': ..., 'a': ..., 'b': ..., 'meta': ...}
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise IOError(f"Could not read file {filepath}: {e}")
    
    required_cols = ['setting_a', 'setting_b', 'outcome_a', 'outcome_b']
    for col in required_cols:
        if col not in df.columns:
            # Try mapping common alternatives
            if 'AliceSetting' in df.columns:
                df.rename(columns={'AliceSetting': 'setting_a', 'BobSetting': 'setting_b',
                                   'AliceOutcome': 'outcome_a', 'BobOutcome': 'outcome_b'}, inplace=True)
            else:
                raise ValueError(f"CSV missing required column: {col}")
                
    # Convert to numpy
    a = df['setting_a'].values.astype(int)
    b = df['setting_b'].values.astype(int)
    
    # Outcomes in BBT are usually 0/1. We need +1/-1 for CHSH S calculation, 
    # but our simulation uses 0 for 'no click'.
    # If the data is from a loophole-free experiment, there are no 'no clicks' (or they are labeled).
    # Let's assume 0/1 are the outcomes. Map 0->+1, 1->-1 (standard spin).
    # Or map 0->-1, 1->+1.
    # Let's check standard convention. usually 0->+1, 1->-1.
    
    # Check if data has -1.
    if df['outcome_a'].min() < 0:
        # Already signed
        x = df['outcome_a'].values.astype(int)
        y = df['outcome_b'].values.astype(int)
    else:
        # Map 0->1, 1->-1
        x = np.where(df['outcome_a'].values == 0, 1, -1)
        y = np.where(df['outcome_b'].values == 0, 1, -1)
        
    # Calculate S
    from ..sims.chsh import estimate_chsh_S
    S = estimate_chsh_S(x, y, a, b)
    
    return {
        'x': x, 'y': y, 'a': a, 'b': b,
        's_value': S,
        'meta': {
            'source': filepath,
            'n_samples': len(df)
        }
    }

def load_nist_binary(filepath: str) -> Dict:
    """
    Load NIST Bell Test binary data (time tags).
    Format appears to be sequence of 64-bit integers (Little Endian).
    Structure (Hypothesis based on hexdump):
    - Word 0: Channel / Event Type
    - Word 1: Timestamp
    - ...
    
    Note: This function parses the raw events. To compute CHSH, one needs
    to coincide these events between Alice and Bob files.
    For this demonstration, we parse the file and return the raw events.
    """
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()
    except Exception as e:
        raise IOError(f"Could not read file {filepath}: {e}")
    
    # Parse as 64-bit integers
    data = np.frombuffer(raw, dtype=np.uint64)
    
    # Reshape? If interleaved.
    # Based on hexdump: 06 00... (Type), 1e 68... (Time)
    # It seems to be pairs.
    if len(data) % 2 != 0:
        # Maybe header?
        pass
        
    # Let's assume it's a stream of events.
    # We return the raw array for analysis.
    
    return {
        'raw_data': data,
        'meta': {
            'source': filepath,
            'size_bytes': len(raw),
            'n_words': len(data)
        }
    }

def mock_big_bell_test_data(N: int = 1000) -> pd.DataFrame:
    """Generate a mock CSV for testing the loader."""
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=N)
    b = rng.integers(0, 2, size=N)
    # Perfect correlations for testing
    # A=0, B=0 -> E=0.7 (same prob 0.85)
    # ...
    # Just use our simulator to generate data and save as CSV
    from ..sims.chsh import simulate_chsh
    angles = (0.0, np.pi/4, np.pi/8, -np.pi/8)
    sim = simulate_chsh(N, angles, {}, {'etaA': 1.0, 'etaB': 1.0})
    
    # Convert -1 to 1, 1 to 0 for "raw" look? Or keep signed.
    # Let's keep signed.
    
    df = pd.DataFrame({
        'setting_a': sim['a'],
        'setting_b': sim['b'],
        'outcome_a': sim['x'],
        'outcome_b': sim['y']
    })
    return df
