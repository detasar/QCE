# Paper 1: TARA - Conformal Prediction for Quantum Certification

## TARA: Test-by-Adaptive-Ranks for Quantum Anomaly Detection with Conformal Prediction Guarantees

### Abstract

Statistical certification of quantum devices requires distinguishing genuine quantum correlations from classical or adversarial alternatives. We introduce **TARA** (Test-by-Adaptive-Ranks), a conformal prediction framework providing distribution-free coverage guarantees for quantum anomaly detection.

TARA operates in two modes:
- **TARA-k** (batch): Tests whether a sample batch originates from the calibration distribution
- **TARA-m** (streaming): Provides anytime-valid p-values via martingale betting

### Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| Coverage guarantee | 1-α | Distribution-free |
| IBM CHSH | 2.725 ± 0.04 | Hardware validated |
| AUC (quantum vs classical) | 0.96 | High discrimination |
| False alarm rate | ≤ α | Guaranteed |
| Security margin | 36% | Above classical bound |

### Experiments

| File | Description |
|------|-------------|
| `tara_detectors.py` | TARA-k and TARA-m detector classes |
| `tara_batch_detection.py` | TARA-k batch anomaly detection experiment |
| `data_utils.py` | Data loading and CHSH computation utilities |

### Usage

```python
from experiments.tara_batch_detection import TARAk

# Load IBM hardware data
data = load_ibm_data('data/ibm_hardware_real.csv')

# Create detector calibrated on product states
tara = TARAk(calibration_data, alpha=0.05)

# Test quantum data
result = tara.test(test_data)
print(f"Detected: {result['detected']}")
print(f"p-value: {result['p_value']:.4f}")
```

### Data

- `ibm_hardware_real.csv`: Real IBM Quantum hardware measurements
  - 10,000+ CHSH correlation measurements
  - CHSH S = 2.725 ± 0.04
  - Collected from ibm_torino (127 qubits)

### Citation

```bibtex
@article{tasar2025tara,
  title={TARA: Test-by-Adaptive-Ranks for Quantum Anomaly Detection
         with Conformal Prediction Guarantees},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint},
  year={2025}
}
```

### Requirements

```
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
qiskit>=1.0.0
matplotlib>=3.7.0
```
