# Quantum Certification Experiments (QCE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/)

This repository contains the complete experimental code, data, and results for a trilogy of papers on quantum certification, adversarial limits, and hardware validation of fundamental quantum constants.

## Papers

### Paper 1: TARA - Conformal Prediction for Quantum Certification
**TARA: Test-by-Adaptive-Ranks for Quantum Anomaly Detection with Conformal Prediction Guarantees**

Statistical framework for certifying quantum correlations using conformal prediction. Introduces TARA-k (batch detection) and TARA-m (streaming martingale) methods with provable coverage guarantees.

📁 [`P1_TARA_Conformal_Quantum/`](./P1_TARA_Conformal_Quantum/)

### Paper 2: Grothendieck Constant Hardware Measurement
**First Multi-Platform Hardware Measurement of the Grothendieck Constant in Quantum Systems**

First systematic measurement of K_G on both superconducting (IBM Quantum) and trapped-ion (IonQ Forte Enterprise) quantum hardware. Establishes K_G = 1.408 ± 0.006 on IonQ (only 0.44% deviation from √2).

📁 [`P2_Grothendieck_DeFinetti_Hardware/`](./P2_Grothendieck_DeFinetti_Hardware/)

### Paper 3: Eve-GAN Adversarial Limits
**Adversarial Limits of Quantum Certification: When Eve Defeats Detection**

Establishes fundamental adversarial limits using Eve-GAN. Key findings: α ≥ 0.95 detection limit, 44-point calibration leakage, S = 2.05 phase transition.

📁 [`P3_Eve_GAN_Adversarial_Limits/`](./P3_Eve_GAN_Adversarial_Limits/)

## Key Results

| Metric | Theory | IBM Quantum | IonQ Forte | Status |
|--------|--------|-------------|------------|--------|
| CHSH S | 2.828 | 2.725 | 2.716 | ✓ Validated |
| K_G | √2 ≈ 1.414 | 1.363 | **1.408** | ✓ 0.44% dev |
| LGI K₃ | 1.5 | 1.38 | 1.365 | ✓ 10.2σ violation |
| GHZ S₃ | 5.66 | 0.51 | **5.514** | ✓ IonQ 97.4% |
| Eve Detection | — | — | α ≥ 0.95 | ⚠️ Limit found |

## Repository Structure

```
QCE/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
│
├── P1_TARA_Conformal_Quantum/            # Paper 1
│   ├── README.md                          # Paper overview
│   ├── experiments/                       # TARA-k, TARA-m detection
│   │   ├── tara_detectors.py              # Core detector classes
│   │   ├── tara_batch_detection.py        # Batch experiment
│   │   └── data_utils.py                  # Data loading utilities
│   └── data/
│       └── ibm_hardware_real.csv          # IBM Quantum data
│
├── P2_Grothendieck_DeFinetti_Hardware/   # Paper 2
│   ├── README.md                          # Paper overview
│   ├── experiments/                       # K_G, LGI, GHZ experiments
│   │   ├── grothendieck_measurement.py    # K_G visibility sweep
│   │   ├── definetti_error.py             # De Finetti analysis
│   │   ├── ionq_chsh.py                   # IonQ CHSH measurement
│   │   ├── ionq_kg_sweep.py               # IonQ K_G sweep
│   │   ├── lgi_measurement.py             # Leggett-Garg test
│   │   └── ghz_svetlichny.py              # GHZ Svetlichny test
│   └── results/                           # QPU results
│
└── P3_Eve_GAN_Adversarial_Limits/        # Paper 3
    ├── README.md                          # Paper overview
    ├── experiments/                       # Eve-GAN experiments
    │   ├── eve_models.py                  # GAN architectures
    │   ├── tara_detectors.py              # Detection algorithms
    │   └── interpolation_analysis.py      # Honest-Eve mixture analysis
    └── data/
        └── ibm_hardware_real.csv          # Training data
```

## Quick Start

### Installation

```bash
git clone https://github.com/detasar/QCE.git
cd QCE
pip install -r requirements.txt
```

### Running Experiments

**Paper 1 - TARA Detection:**
```bash
cd P1_TARA_Conformal_Quantum/experiments
python tara_batch_detection.py --shots 8192
```

**Paper 2 - Grothendieck Measurement:**
```bash
cd P2_Grothendieck_DeFinetti_Hardware/experiments
python grothendieck_measurement.py --platform ionq
```

**Paper 3 - Eve-GAN Analysis:**
```bash
cd P3_Eve_GAN_Adversarial_Limits/experiments
python interpolation_analysis.py --runs 15
```

## Hardware Platforms

### IBM Quantum
- **Devices**: ibm_torino (127 qubits), ibm_fez (156 qubits)
- **Gate fidelity**: ~99.9% (1Q), ~99% (2Q)
- **Coherence**: T₁ ≈ 300μs, T₂ ≈ 100μs

### IonQ Forte Enterprise
- **Qubits**: 36 algorithmic (¹⁷¹Yb⁺ trapped ions)
- **Connectivity**: All-to-all
- **Gate fidelity**: >99.5% (1Q), >99.0% (2Q)
- **Coherence**: T₁ > 10s, T₂ ≈ 1s

## Citation

If you use this code, please cite our papers:

```bibtex
@article{tasar2025tara,
  title={TARA: Test-by-Adaptive-Ranks for Quantum Anomaly Detection
         with Conformal Prediction Guarantees},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}

@article{tasar2025grothendieck,
  title={First Multi-Platform Hardware Measurement of the Grothendieck
         Constant in Quantum Systems},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}

@article{tasar2025evegan,
  title={Adversarial Limits of Quantum Certification:
         When Eve Defeats Detection},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint arXiv:2501.xxxxx},
  year={2025}
}
```

## Author

**Davut Emre Taşar**
Independent Researcher, Madrid, Spain
📧 detasar@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- IBM Quantum for hardware access
- Microsoft Azure Quantum for IonQ access
- Open-source communities: PyTorch, Qiskit, scikit-learn
