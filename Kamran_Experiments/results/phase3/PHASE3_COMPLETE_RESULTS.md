# Phase 3: Advanced QKD Analysis - Complete Results

**Author**: Davut Emre Tasar
**Date**: December 2024
**Last Updated**: December 2025 (Post-Audit)
**Experiments**: 3.1 - 3.18 (11 experiments validated)
**Status**: 11/11 PASS (ALL EXPERIMENTS VALIDATED)

---

## Executive Summary

Phase 3 explored advanced QKD security topics across 6 categories:

| Category | Experiments | Status | Key Finding |
|----------|-------------|--------|-------------|
| A: Detection | exp_3_1 - 3_6 | PASS | MMD outperforms uniformity tests |
| B: Finite Key | exp_3_7 | PASS | 64.6% convergence at n=1M |
| C: Channel | exp_3_8 | PASS | 95.3 km max secure distance |
| D: Attacks | exp_3_10 | PASS | Devetak-Winter bound verified |
| E: ML Detection | exp_3_15 | PASS | AUC-ROC = 0.9796 |
| F: Optimization | exp_3_18 | PASS | 8.4% key rate improvement |

---

## Category A: Detection Methods (exp_3_1 - exp_3_6)

### Critical Finding: P-Value Non-Uniformity

TARA conformal p-values from discrete QKD are **fundamentally non-uniform**:
- KS test p-value = 2.35e-23 (Phase 1)
- KS test p-value = 3.84e-36 (Phase 3)

### What Works vs What Doesn't

| Method | FPR | Attack Detection | Recommendation |
|--------|-----|------------------|----------------|
| **TARA-MMD** | ~5% | 7/7 attacks | **RECOMMENDED** |
| **Wasserstein** | 0% | 5/5 attacks | **RECOMMENDED** |
| CUSUM | 6.7% | Streaming OK | Use with caution |
| KS (uniformity) | 100% | N/A | **DO NOT USE** |
| Anderson-Darling | 100% | N/A | **DO NOT USE** |
| Chi-squared | 96.7% | N/A | **DO NOT USE** |

### Key Insight
Use **two-sample tests** (MMD, Wasserstein) that compare calibration vs test distributions.
Avoid **uniformity tests** (KS, AD, CvM) that assume p-values ~ Uniform(0,1).

---

## Category B: Finite-Key Security (exp_3_7)

### Purpose
Compute secure key rates accounting for statistical fluctuations in finite sample sizes.

### Key Results

| Sample Size (n) | Finite Rate | Asymptotic Rate | Ratio |
|-----------------|-------------|-----------------|-------|
| 1,000 | 0 | 0.806 | 0% |
| 10,000 | 0 | 0.806 | 0% |
| 50,000 | 0.308 | 0.806 | 38% |
| 100,000 | 0.390 | 0.806 | 48% |
| 1,000,000 | 0.521 | 0.806 | 65% |

### QBER Security Threshold

| QBER | Secure? | Key Rate | Key Length (n=100k) |
|------|---------|----------|---------------------|
| 1% | YES | 0.650 | 55,249 bits |
| 3% | YES | 0.390 | 33,107 bits |
| 5% | YES | 0.186 | 15,781 bits |
| 7% | YES | 0.012 | 1,048 bits |
| 9% | NO | 0 | 0 |
| 11% | NO | 0 | 0 |

### Security Parameter Impact

| epsilon | Finite Rate | Key Length |
|---------|-------------|------------|
| 1e-6 | 0.311 | 13,196 |
| 1e-8 | 0.267 | 11,365 |
| 1e-10 | 0.229 | 9,737 |
| 1e-12 | 0.194 | 8,255 |
| 1e-15 | 0.147 | 6,232 |

### Minimum Samples for Key Generation

| Target Key | Min Sifted Bits | QBER=3% |
|------------|-----------------|---------|
| 256 bits | 12,631 | Achieved |
| 1,024 bits | 15,280 | Achieved |
| 4,096 bits | 25,084 | Achieved |

---

## Category C: Fiber Channel Models (exp_3_8)

### Channel Parameters
- Fiber attenuation: 0.2 dB/km @ 1550nm
- Connector loss: 0.3 dB each
- Detector efficiency: 10% (InGaAs APD)

### Distance vs Loss

| Distance | Fiber Loss | Total Loss (incl. detector) |
|----------|------------|----------------------------|
| 10 km | 2.0 dB | 12.6 dB |
| 50 km | 10.0 dB | 20.6 dB |
| 100 km | 20.0 dB | 30.6 dB |
| 150 km | 30.0 dB | 40.6 dB |
| 200 km | 40.0 dB | 50.6 dB |

### Maximum Secure Distance
- **Standard InGaAs detectors**: 95.3 km
- **SNSPD detectors** (90% efficiency): Extends range significantly

### QBER Components at 100 km

| Source | QBER Contribution |
|--------|-------------------|
| Dark counts | 0.115% |
| Polarization drift | 1.0% |
| Afterpulse | 0.5% |
| **Total channel** | 1.6% |
| Intrinsic (source) | 1.0% |
| **TOTAL** | ~2.6% |

---

## Category D: Collective Attacks (exp_3_10)

### Attack Types Analyzed

| Attack | QBER Introduced | Eve's Information | Key Rate |
|--------|-----------------|-------------------|----------|
| Optimal Cloning (F=5/6) | 16.7% | 0.65 bits | 0 |
| Intercept-Resend 10% | 2.5% | 0.05 bits | 0.78 |
| Intercept-Resend 50% | 12.5% | 0.25 bits | 0.21 |
| Beam Splitter 20% | 1.9% | 0.05 bits | 0.82 |
| Phase Remapping 20% | 5.0% | 0.10 bits | 0.61 |
| Trojan Horse | 0% | Variable | Undetectable via QBER |

### Devetak-Winter Bound Verification
All attacks correctly respect: r = max(0, I(A:B) - χ(Eve))

### Detection Probabilities (100 sample bits)

| Attack | QBER | Detection Probability |
|--------|------|----------------------|
| Optimal Cloning | 16.7% | 99.99999% |
| Intercept-Resend 20% | 5.0% | 99.4% |
| Intercept-Resend 40% | 10.0% | 99.997% |
| Trojan Horse | 0% | **0%** (Undetectable) |

### Key Insight
Trojan horse attacks are **undetectable via QBER**. Requires separate countermeasures (optical isolation, monitoring).

---

## Category E: ML-Based Detection (exp_3_15)

### Isolation Forest Detector

**Model**: Unsupervised anomaly detection
**Features**: 18 (CHSH, correlators, QBER, entropy, visibility, etc.)

### Performance Metrics

| Metric | Value |
|--------|-------|
| AUC-ROC | **0.9796** |
| FPR | 3.0% |
| Training samples | 200 |

### Attack Detection Rates

| Attack Type | Strength | Detection Rate |
|-------------|----------|----------------|
| Intercept-Resend | 30% | 44% |
| Decorrelation | 40% | 46% |
| Visibility | 50% | 72% |

### Feature Importance (by anomaly score correlation)

| Rank | Feature | Correlation |
|------|---------|-------------|
| 1 | entropy | +0.39 |
| 2 | CHSH | -0.25 |
| 3 | high_pval | -0.13 |
| 4 | low_pval | -0.13 |
| 5 | E01 | +0.12 |

### Contamination Impact

| Contamination | FPR | TPR |
|---------------|-----|-----|
| 1% | 0% | 30% |
| 5% | 2% | 44% |
| 10% | 6% | 68% |
| 15% | 10% | 76% |

---

## Category F: Adaptive Sampling (exp_3_18)

### Strategy
Start with high sampling rate (25%), reduce as confidence increases.

### Rate Adaptation

| QBER | Initial Rate | Final Rate | Behavior |
|------|--------------|------------|----------|
| 1% | 25% | 5% | Fast reduction (low QBER) |
| 3% | 25% | 11% | Moderate reduction |
| 5% | 25% | 11% | Moderate reduction |
| 8% | 25% | 25% | Maintain high (near threshold) |
| 10% | 25% | 25% | Maintain high (near threshold) |

### Performance Comparison

| Metric | Adaptive | Fixed (15%) | Improvement |
|--------|----------|-------------|-------------|
| Avg Sampled | 1,555 | 1,503 | +3.5% |
| Raw Key Bits | 45,770 | 42,491 | +7.7% |
| Final Key Bits | 21,940 | 20,248 | **+8.4%** |

### Precision Convergence

| Samples | CI Width |
|---------|----------|
| 100 | 0.102 |
| 300 | 0.055 |
| 500 | 0.044 |
| 700 | 0.035 |
| 1000 | 0.030 |

---

## Key Insights for Papers

### 1. P-Value Non-Uniformity is Fundamental
TARA conformal p-values for discrete QKD are inherently non-uniform.
**Solution**: Use two-sample tests (MMD, Wasserstein) instead of uniformity tests.

### 2. Finite-Key Penalty is Significant
At n=1M with epsilon=1e-6, finite-key rate is ~65% of asymptotic.
For practical security (epsilon=1e-10), need 10M+ sifted bits for ~50% efficiency.

### 3. Fiber QKD Limited to ~100 km with Standard Detectors
Maximum secure distance with InGaAs APDs: 95 km.
SNSPD can extend this significantly.

### 4. Trojan Horse Attacks are Invisible to QBER
Side-channel attacks leave no QBER signature.
Requires hardware countermeasures.

### 5. ML Detection Achieves 98% AUC
Isolation Forest with 18 features provides excellent attack discrimination.
Entropy and CHSH are most informative features.

### 6. Adaptive Sampling Improves Key Rate by 8%
Dynamic sampling based on confidence yields more key bits
without compromising security.

---

## Validation Summary

| Experiment | Tests | Passed | Status |
|------------|-------|--------|--------|
| exp_3_7 (Finite Key) | 5 | 5 | PASS |
| exp_3_8 (Fiber) | 5 | 5 | PASS |
| exp_3_10 (Attacks) | 6 | 6 | PASS |
| exp_3_15 (ML) | 6 | 6 | PASS |
| exp_3_18 (Adaptive) | 6 | 6 | PASS |
| **TOTAL** | 28 | 28 | **ALL PASS** |

---

## Recommendations

### For Implementation
1. Use MMD for attack detection (not KS/AD/CvM)
2. Target n > 50,000 sifted bits for practical key generation
3. Use SNSPD detectors for distances > 50 km
4. Implement optical isolation against Trojan horse
5. Use adaptive sampling for efficiency gains

### For Papers
1. Document p-value non-uniformity limitation
2. Report realistic finite-key rates (not asymptotic)
3. Include side-channel attack considerations
4. Use proper two-sample tests for comparisons

---

**Phase 3 Complete** - December 2024
*Last validated: December 2025 (Post-Audit) - 11/11 PASS*
*All experiments validated with corrected methodology (p-value non-uniformity issue addressed)*
