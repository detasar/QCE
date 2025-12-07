"""
Kamran QKD Experiments - Utility Modules

This package provides core utilities for the QKD experiment pipeline:
- circuits: Bell state and measurement circuits
- authenticated_channel: HMAC-based classical channel authentication
- sifting: Key sifting protocol
- error_correction: CASCADE protocol
- privacy_amp: Toeplitz hashing for privacy amplification
- security_analysis: CHSH and device-independent bounds
- tara_integration: TARA-k and TARA-m wrappers

Author: Davut Emre Tasar
Date: December 2024
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PHASE1_RESULTS = RESULTS_DIR / "phase1"
PHASE2_RESULTS = RESULTS_DIR / "phase2"
PHASE3_RESULTS = RESULTS_DIR / "phase3"

# Ensure directories exist
for d in [RESULTS_DIR, PHASE1_RESULTS, PHASE2_RESULTS, PHASE3_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# IBM API Key path (set via environment variable or config file)
import os
IBM_API_KEY_PATH = Path(os.environ.get('IBM_QUANTUM_API_KEY_PATH', 'configs/apikey.json'))

# TARA module path (relative to project)
import sys
TARA_PATH = PROJECT_ROOT.parent / 'experiments'
if TARA_PATH.exists() and str(TARA_PATH) not in sys.path:
    sys.path.insert(0, str(TARA_PATH))
