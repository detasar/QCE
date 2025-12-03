# Paper 3: Eve-GAN Adversarial Limits

## Impossibility of Undetectable Eavesdropping: GANs Cannot Mimic Quantum Correlations

### Abstract

We demonstrate fundamental limits on adversarial attacks against quantum communication protocols using Generative Adversarial Networks (GANs). Despite training Eve-GAN to maximize Bell violation and minimize distribution divergence, the generated correlations remain statistically distinguishable from genuine quantum data using conformal prediction techniques.

### Key Results

| Metric | Value | Significance |
|--------|-------|--------------|
| **TARA-k AUC** | 0.93 | Near-perfect separation |
| **Detection at α=0.95** | 100% | All Eve samples detected |
| **α-threshold** | 0.95 | TARA breaks down below 5% Eve |
| **Eve CHSH** | 2.5 | Matches quantum violation |
| **KL Divergence** | 0.08 | Low distributional error |

### The Fundamental Limit

Despite Eve-GAN achieving:
- Correct CHSH violation (S = 2.5)
- Low KL divergence to IBM data
- Matching marginal distributions

TARA detection remains possible because:
1. **Higher-order correlations**: GAN captures P(a,b|x,y) but not temporal/contextual structure
2. **44-point calibration**: Dense p-value distribution reveals subtle anomalies
3. **Conformal guarantee**: Type-I error controlled at any α level

### Experiments

| File | Description |
|------|-------------|
| `eve_models.py` | Eve-GAN neural network architectures (V1 and V2) |
| `tara_detectors.py` | TARA-k and TARA-m detection algorithms |
| `interpolation_analysis.py` | Honest-Eve mixture detection threshold analysis |

### Eve-GAN Architecture

```python
class EveGenerator(nn.Module):
    """
    Generator outputs P(a,b|x,y) for quantum outcomes.

    Input: latent (16) + x_onehot (2) + y_onehot (2) = 20
    Output: 4-dim softmax (joint probabilities)
    """

class EveGeneratorV2(nn.Module):
    """
    Improved version with setting-specific heads.
    Better captures anti-correlation in (1,1) setting.
    """
```

### Detection Methodology

**TARA-k (Batch Detection)**:
- Calibrate on product state (classical) data
- Compute conformal p-values for test batch
- KS test against Uniform(0,1)
- Threshold: KS > 0.2 indicates Eve

**TARA-m (Streaming Detection)**:
- Martingale-based sequential test
- Wealth grows exponentially under attack
- Early detection possible

### Interpolation Analysis

Key finding: Detection threshold at α = 0.95 (5% Eve contamination)

| α (Honest %) | Detection Rate | CHSH |
|--------------|----------------|------|
| 1.00 | 0% | 2.73 |
| 0.95 | 15% | 2.72 |
| 0.90 | 45% | 2.70 |
| 0.80 | 90% | 2.66 |
| 0.50 | 100% | 2.62 |
| 0.00 | 100% | 2.50 |

### PR-Box Analysis

The PR-Box (Popescu-Rohrlich box) achieves S = 4 (algebraic maximum) but violates no-signaling only at the boundary. Our analysis shows:

- Tsirelson bound S = 2√2 emerges naturally from learnability constraints
- Super-quantum correlations (2√2 < S < 4) are strictly detectable
- CSO (Conformal Sheaf Obstruction) correlates with Bell violation

### Usage

```python
from experiments.eve_models import EveGenerator, generate_eve_samples, train_eve_gan
from experiments.tara_detectors import TARAk

# Train Eve-GAN on IBM data
generator, history = train_eve_gan(ibm_data, epochs=2000, target_S=2.5)

# Generate Eve samples
eve_data = generate_eve_samples(generator, n_samples=10000)

# Detection with TARA
tara = TARAk(calibration_data)
result = tara.test(eve_data)
print(f"Detected: {result['detected']}, KS: {result['ks_statistic']:.4f}")
```

### Hardware Validation

Eve-GAN was trained on real IBM Quantum hardware data:
- **Backend**: ibm_torino (127 qubits)
- **Bell State**: |Φ+⟩ = (|00⟩ + |11⟩)/√2
- **CHSH**: 2.725 (96.3% of Tsirelson)
- **Samples**: 16,000 per setting (64,000 total)

### Citation

```bibtex
@article{tasar2025evegan,
  title={Impossibility of Undetectable Eavesdropping: GANs Cannot
         Mimic Quantum Correlations},
  author={Ta{\c{s}}ar, Davut Emre},
  journal={arXiv preprint},
  year={2025}
}
```

### Requirements

```
numpy>=1.24.0
scipy>=1.11.0
torch>=2.0.0
pandas>=2.0.0
matplotlib>=3.7.0
qiskit>=1.0.0
```
