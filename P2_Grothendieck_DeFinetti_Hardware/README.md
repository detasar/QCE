# Paper 2: Grothendieck Constant Hardware Measurement

## First Multi-Platform Hardware Measurement of the Grothendieck Constant in Quantum Systems

### Abstract

The Grothendieck constant K_G connects classical optimization to quantum correlations through the Tsirelson bound. For the CHSH scenario, K_G = √2 ≈ 1.4142. We present the first systematic multi-platform hardware measurement of K_G on both superconducting (IBM Quantum) and trapped-ion (IonQ Forte Enterprise) quantum computers.

### Key Results

| Platform | K_G Measured | Deviation from √2 | Status |
|----------|-------------|-------------------|--------|
| **IonQ Forte Enterprise** | 1.408 ± 0.006 | **0.44%** | Best result |
| IBM Torino | 1.363 ± 0.012 | 3.6% | Hardware validated |
| Theory | √2 = 1.4142 | — | Reference |

### Additional Measurements

| Experiment | IonQ Result | Significance |
|------------|-------------|--------------|
| CHSH S | 2.716 | 96% of Tsirelson |
| GHZ Svetlichny S₃ | 5.514 | 97.4% of quantum max |
| LGI K₃ | 1.365 ± 0.036 | **10.2σ violation** |
| Bell Fidelity | 0.984 | Near-ideal |

### Experiments

| File | Description |
|------|-------------|
| `grothendieck_measurement.py` | K_G visibility sweep measurement |
| `definetti_error.py` | De Finetti error vs CHSH analysis |
| `ionq_chsh.py` | IonQ Forte CHSH measurement |
| `ionq_kg_sweep.py` | IonQ K_G comprehensive sweep |
| `lgi_measurement.py` | Leggett-Garg inequality test |
| `ghz_svetlichny.py` | Three-qubit GHZ Svetlichny test |

### The Grothendieck-Tsirelson Connection

The Grothendieck constant relates maximum classical and quantum correlations:

```
K_G = S_max^quantum / S_max^classical = 2√2 / 2 = √2 ≈ 1.4142
```

We measure this by computing:

```python
K_G = CHSH(v) / (2 * v)
```

where `v` is the effective visibility of the Bell state.

### Usage

```python
from experiments.grothendieck_measurement import measure_kg

# Run K_G measurement on simulated data
result = measure_kg(visibilities=[0.1, 0.2, ..., 1.0])
print(f"K_G = {result['kg']:.4f} ± {result['kg_error']:.4f}")
```

### Hardware Platforms

**IonQ Forte Enterprise (Azure Quantum)**
- 36 algorithmic qubits (¹⁷¹Yb⁺ trapped ions)
- All-to-all connectivity
- T₂ ≈ 1 second coherence
- Gate fidelity > 99%

**IBM Quantum (Torino, Fez)**
- 127-156 transmon qubits
- Heavy-hex topology
- T₂ ≈ 100 μs coherence
- Gate fidelity ~99%

### De Finetti Error Relationship

We establish the empirical relationship:

```
ε = 0.498 × S - 0.468  (for v > 1/3)
```

where ε is the De Finetti error and S is the CHSH value. This provides a direct mapping from Bell violation to entanglement quantification.

### Citation

```bibtex
@article{tasar2025grothendieck,
  title={First Multi-Platform Hardware Measurement of the Grothendieck
         Constant in Quantum Systems},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint},
  year={2025}
}
```

### Requirements

```
numpy>=1.24.0
scipy>=1.11.0
qiskit>=1.0.0
azure-quantum>=2.0.0
matplotlib>=3.7.0
```
