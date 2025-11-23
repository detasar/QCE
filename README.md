# Quantum Conformal Entanglement (QCE)

**Conformal Prediction for Quantum Entanglement Detection**

This repository contains the complete source code for the research paper *"Conformal Prediction for Quantum Entanglement: Robustness and Detection"* by Davut Emre Taşar.

## Overview

This project applies **Conformal Prediction (CP)** to quantum entanglement certification, demonstrating:
- CP's robustness to quantum contextuality in the CHSH scenario
- LHV-calibrated anomaly detection (ROC AUC ≈ 0.96)
- No-go theorem for product-threshold Conformal Entanglement Witnesses (CEW)
- Validation on IBM quantum hardware

## Installation

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── src/                   # Core QCE library
│   ├── sims/             # CHSH simulations (quantum & LHV models)
│   ├── cp/               # Conformal prediction implementations
│   ├── witness/          # Entanglement witnesses (CEW, CMI, ML)
│   ├── stats/            # Statistical tools
│   ├── viz/              # Visualization utilities
│   └── cli.py            # Command-line interface
├── experiments/          # Reproducible experiments
│   ├── run_tara.py      # Main experiment runner (TARA programs A-F)
│   └── configs/         # Experiment configurations (JSON/YAML)
└── tests/                # Unit tests
```

## Usage

### Running Experiments

Execute the main TARA experiments:

```bash
python experiments/run_tara.py --config experiments/configs/tara_A.json
```

Available configurations:
- `tara_A.json` - LHV Atlas Generation
- `tara_B.json` - Sheaf-Index Analysis
- `tara_C.json` - CEW Threshold Sweeps
- `tara_D.json` - Anomaly Detection Benchmarks
- `tara_E.json` - Sensitivity Analysis
- `tara_F.json` - Martingale Robustness Tests

### Running Tests

```bash
pytest tests/
```

## Citation

If you use this code, please cite:

```
@misc{tasar2025qce,
  author = {Taşar, Davut Emre},
  title = {Conformal Prediction for Quantum Entanglement: Robustness and Detection},
  year = {2025},
  url = {https://github.com/detasar/QCE}
}
```

## License

This project is released under a **Custom Non-Commercial License**. See [`LICENSE`](LICENSE) for details.

**Attribution:** Davut Emre Taşar (detasar@gmail.com)  
**Commercial Use:** Requires explicit written permission.

## AI Disclosure

This research utilized generative AI tools (GPT-5.1, Gemini 3, Claude Sonnet 4.5) for writing assistance, coding support, and literature research. All scientific claims and experimental designs were verified by the author.

## Contact

**Davut Emre Taşar**  
Independent Researcher  
Email: detasar@gmail.com  
GitHub: https://github.com/detasar/QCE
