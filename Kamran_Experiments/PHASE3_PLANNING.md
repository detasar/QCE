# Phase 3: Advanced Simulations - Comprehensive Planning

> **Version**: 1.0
> **Date**: 2024-12-07
> **Author**: Davut Emre Tasar
> **Status**: PLANNING
> **Prerequisite**: Phase 1 & 2 completed successfully

---

## Executive Summary

Phase 3 extends our QKD simulation framework with advanced capabilities:

1. **Enhanced Statistical Detection** - Alternative tests to complement/replace TARA
2. **Finite-Key Security** - Moving from asymptotic to practical security proofs
3. **Channel Modeling** - Realistic fiber and free-space effects
4. **Advanced Attacks** - Collective, coherent, and side-channel attacks
5. **Machine Learning** - Neural network-based anomaly detection
6. **Protocol Optimizations** - Adaptive sampling, alternative EC

**Total Experiments**: 25 experiments across 6 categories
**Estimated Duration**: 15-20 days of development/testing

---

## Category A: Enhanced Statistical Detection (TARA Alternatives)

### Motivation

Phase 1-2 results showed:
- TARA-K (KS statistic): Detected 0/4 attacks - **insufficient sensitivity**
- TARA-M (Martingale): Detected 3/4 attacks - **good but has false positives**
- CHSH correlation: Detected 4/4 attacks - **gold standard**

We need alternative statistical tests that:
1. Are more sensitive than TARA-K
2. Have lower false positive rate than TARA-M
3. Provide complementary detection mechanisms
4. Support real-time/streaming operation

---

### Experiment 3.1: TARA-W (Wasserstein Distance)

**File**: `exp_3_1_tara_wasserstein.py`

**Concept**: Wasserstein (Earth Mover's) distance measures the minimum "work" to transform one distribution into another. More sensitive to shape differences than KS.

**Mathematical Foundation**:
```
W_p(P, Q) = (inf_{γ ∈ Γ(P,Q)} ∫ ||x-y||^p dγ(x,y))^(1/p)

For 1D distributions with CDFs F and G:
W_1(F, G) = ∫_{-∞}^{∞} |F(x) - G(x)| dx
```

**Implementation**:
```python
from scipy.stats import wasserstein_distance
import numpy as np

class TARAWasserstein:
    """
    TARA detector using Wasserstein distance.

    Advantages over KS:
    - Considers magnitude of differences, not just max
    - More sensitive to distribution shape changes
    - Metrizes weak convergence
    """

    def __init__(self, calibration_pvalues: np.ndarray):
        """
        Args:
            calibration_pvalues: P-values from calibration (honest) data
        """
        self.calibration = np.sort(calibration_pvalues)
        self.n_cal = len(calibration_pvalues)

        # Bootstrap threshold estimation
        self.threshold = self._estimate_threshold()

    def _estimate_threshold(self, n_bootstrap: int = 1000,
                            alpha: float = 0.05) -> float:
        """Estimate detection threshold via bootstrap."""
        distances = []
        for _ in range(n_bootstrap):
            # Resample calibration data
            sample = np.random.choice(self.calibration,
                                      size=self.n_cal, replace=True)
            d = wasserstein_distance(self.calibration, sample)
            distances.append(d)

        return np.percentile(distances, 100 * (1 - alpha))

    def test(self, test_pvalues: np.ndarray) -> dict:
        """
        Test for anomaly using Wasserstein distance.

        Returns:
            Dictionary with distance, threshold, and detection result
        """
        distance = wasserstein_distance(self.calibration, test_pvalues)
        detected = distance > self.threshold

        # Effect size (normalized distance)
        effect_size = distance / np.std(self.calibration)

        return {
            'wasserstein_distance': distance,
            'threshold': self.threshold,
            'effect_size': effect_size,
            'detected': detected,
            'confidence': min(1.0, distance / self.threshold)
        }

    def streaming_test(self, new_pvalue: float,
                       window_size: int = 100) -> dict:
        """
        Streaming version with sliding window.
        """
        # Implementation for real-time monitoring
        pass
```

**Test Scenarios**:

| Scenario | Calibration | Test | Expected W |
|----------|-------------|------|------------|
| Honest | U[0,1] | U[0,1] | ~0.01 |
| Weak attack | U[0,1] | Beta(1.2, 1) | ~0.05 |
| Strong attack | U[0,1] | Beta(2, 1) | ~0.15 |
| Severe attack | U[0,1] | Beta(5, 1) | ~0.30 |

**Validation Criteria**:
- [ ] W ≈ 0 for identical distributions
- [ ] W increases monotonically with attack strength
- [ ] Detection rate > TARA-K for all attack types
- [ ] False positive rate < 10%

---

### Experiment 3.2: TARA-MMD (Maximum Mean Discrepancy)

**File**: `exp_3_2_tara_mmd.py`

**Concept**: MMD embeds distributions into a Reproducing Kernel Hilbert Space (RKHS) and measures distance there. Powerful for detecting subtle distributional differences.

**Mathematical Foundation**:
```
MMD²(P, Q) = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]

where k is a kernel function (typically RBF):
k(x,y) = exp(-||x-y||² / (2σ²))
```

**Implementation**:
```python
import numpy as np
from typing import Tuple

class TARAMMD:
    """
    TARA detector using Maximum Mean Discrepancy.

    Based on Gretton et al. (2012) - "A Kernel Two-Sample Test"

    Advantages:
    - Detects any distributional difference (not just location/scale)
    - Works well in high dimensions
    - Has rigorous statistical guarantees
    """

    def __init__(self, calibration_data: np.ndarray,
                 kernel: str = 'rbf',
                 sigma: float = None):
        """
        Args:
            calibration_data: Reference data (n_samples, n_features)
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            sigma: RBF kernel bandwidth (auto-estimated if None)
        """
        self.calibration = calibration_data
        self.kernel = kernel

        if sigma is None:
            # Median heuristic for bandwidth selection
            self.sigma = self._median_heuristic(calibration_data)
        else:
            self.sigma = sigma

        # Pre-compute calibration kernel matrix
        self.K_cal = self._compute_kernel_matrix(calibration_data,
                                                  calibration_data)

    def _median_heuristic(self, X: np.ndarray) -> float:
        """Estimate kernel bandwidth using median heuristic."""
        from scipy.spatial.distance import pdist
        distances = pdist(X.reshape(-1, 1) if X.ndim == 1 else X)
        return np.median(distances)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix."""
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y

        # ||x - y||² = ||x||² + ||y||² - 2<x,y>
        X_sqnorm = np.sum(X**2, axis=1, keepdims=True)
        Y_sqnorm = np.sum(Y**2, axis=1, keepdims=True)

        sq_dist = X_sqnorm + Y_sqnorm.T - 2 * np.dot(X, Y.T)
        return np.exp(-sq_dist / (2 * self.sigma**2))

    def _compute_kernel_matrix(self, X: np.ndarray,
                                Y: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        if self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        elif self.kernel == 'linear':
            X = X.reshape(-1, 1) if X.ndim == 1 else X
            Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
            return np.dot(X, Y.T)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def compute_mmd_squared(self, test_data: np.ndarray) -> float:
        """
        Compute unbiased MMD² estimate.

        MMD²_u = 1/(m(m-1)) Σ_{i≠j} k(xi,xj)
               + 1/(n(n-1)) Σ_{i≠j} k(yi,yj)
               - 2/(mn) Σ_ij k(xi,yj)
        """
        m = len(self.calibration)
        n = len(test_data)

        # K_XX (calibration-calibration)
        K_XX = self.K_cal
        term1 = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1))

        # K_YY (test-test)
        K_YY = self._compute_kernel_matrix(test_data, test_data)
        term2 = (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1))

        # K_XY (calibration-test)
        K_XY = self._compute_kernel_matrix(self.calibration, test_data)
        term3 = 2 * np.sum(K_XY) / (m * n)

        return term1 + term2 - term3

    def permutation_test(self, test_data: np.ndarray,
                         n_permutations: int = 1000) -> Tuple[float, float]:
        """
        Compute p-value via permutation test.

        Returns:
            (mmd_squared, p_value)
        """
        observed_mmd = self.compute_mmd_squared(test_data)

        # Pool all data
        pooled = np.concatenate([self.calibration, test_data])
        m = len(self.calibration)

        null_mmds = []
        for _ in range(n_permutations):
            perm = np.random.permutation(len(pooled))
            X_perm = pooled[perm[:m]]
            Y_perm = pooled[perm[m:]]

            # Temporarily update calibration for MMD computation
            K_XX_perm = self._compute_kernel_matrix(X_perm, X_perm)
            K_YY_perm = self._compute_kernel_matrix(Y_perm, Y_perm)
            K_XY_perm = self._compute_kernel_matrix(X_perm, Y_perm)

            term1 = (np.sum(K_XX_perm) - np.trace(K_XX_perm)) / (m * (m - 1))
            term2 = (np.sum(K_YY_perm) - np.trace(K_YY_perm)) / (len(Y_perm) * (len(Y_perm) - 1))
            term3 = 2 * np.sum(K_XY_perm) / (m * len(Y_perm))

            null_mmds.append(term1 + term2 - term3)

        p_value = np.mean(np.array(null_mmds) >= observed_mmd)

        return observed_mmd, p_value

    def test(self, test_data: np.ndarray, alpha: float = 0.05) -> dict:
        """
        Full MMD test with p-value.
        """
        mmd_sq, p_value = self.permutation_test(test_data)

        return {
            'mmd_squared': mmd_sq,
            'mmd': np.sqrt(max(0, mmd_sq)),
            'p_value': p_value,
            'detected': p_value < alpha,
            'sigma': self.sigma
        }
```

**Test Scenarios**:

| Scenario | Distribution Shift | Expected MMD² |
|----------|-------------------|---------------|
| No attack | None | ~0.001 |
| Location shift | μ + 0.1 | ~0.01 |
| Scale change | σ × 1.5 | ~0.02 |
| Shape change | Normal → Laplace | ~0.03 |
| Mixture | 0.9N + 0.1U | ~0.05 |

**Validation Criteria**:
- [ ] Type I error ≤ α (calibrated)
- [ ] Power > 80% for effect size > 0.5
- [ ] Detects attacks missed by TARA-K
- [ ] Computational efficiency < 1s for n=1000

---

### Experiment 3.3: Sequential Tests (CUSUM & SPRT)

**File**: `exp_3_3_sequential_tests.py`

**Concept**: Sequential tests make decisions as data arrives, minimizing samples needed for detection. Critical for real-time QKD monitoring.

**CUSUM (Cumulative Sum)**:
```
S_n = max(0, S_{n-1} + (X_n - μ_0) - k)

Alarm when S_n > h

Parameters:
- μ_0: Expected mean under H_0
- k: Allowance (typically 0.5 × expected shift)
- h: Threshold (controls ARL₀)
```

**SPRT (Sequential Probability Ratio Test)**:
```
Λ_n = Σ log(f_1(X_i) / f_0(X_i))

Accept H_0 if Λ_n < A
Accept H_1 if Λ_n > B
Continue otherwise

A = log(β / (1-α))
B = log((1-β) / α)
```

**Implementation**:
```python
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

class DecisionState(Enum):
    CONTINUE = "continue"
    ACCEPT_H0 = "accept_h0"  # No attack
    REJECT_H0 = "reject_h0"  # Attack detected

@dataclass
class SequentialTestResult:
    """Result of sequential test at current step."""
    step: int
    statistic: float
    decision: DecisionState
    p_value_estimate: Optional[float] = None


class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) detector for QKD anomaly detection.

    Reference: Page (1954), "Continuous Inspection Schemes"

    Monitors for upward shift in mean of p-values.
    Under H_0 (no attack): p-values ~ U[0,1], mean = 0.5
    Under H_1 (attack): p-values shift toward 0 or 1
    """

    def __init__(self, target_mean: float = 0.5,
                 allowance: float = 0.05,
                 threshold: float = 5.0,
                 two_sided: bool = True):
        """
        Args:
            target_mean: Expected mean under H_0 (0.5 for uniform p-values)
            allowance: Slack parameter k (typically 0.5 × expected shift)
            threshold: Decision threshold h (higher = fewer false alarms)
            two_sided: Monitor both directions
        """
        self.mu_0 = target_mean
        self.k = allowance
        self.h = threshold
        self.two_sided = two_sided

        # State
        self.S_plus = 0.0  # Upper CUSUM
        self.S_minus = 0.0  # Lower CUSUM
        self.step = 0
        self.history = []

    def reset(self):
        """Reset detector state."""
        self.S_plus = 0.0
        self.S_minus = 0.0
        self.step = 0
        self.history = []

    def update(self, x: float) -> SequentialTestResult:
        """
        Process one observation.

        Args:
            x: New observation (p-value)

        Returns:
            SequentialTestResult with current status
        """
        self.step += 1

        # Update CUSUM statistics
        self.S_plus = max(0, self.S_plus + (x - self.mu_0) - self.k)

        if self.two_sided:
            self.S_minus = max(0, self.S_minus - (x - self.mu_0) - self.k)
            statistic = max(self.S_plus, self.S_minus)
        else:
            statistic = self.S_plus

        self.history.append(statistic)

        # Decision
        if statistic > self.h:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=statistic,
            decision=decision
        )

    def get_run_length(self) -> Optional[int]:
        """Get step at which alarm was raised, or None."""
        for i, s in enumerate(self.history):
            if s > self.h:
                return i + 1
        return None


class SPRTDetector:
    """
    Sequential Probability Ratio Test for QKD.

    Reference: Wald (1945), "Sequential Tests of Statistical Hypotheses"

    Tests H_0: p-values ~ U[0,1] vs H_1: p-values ~ Beta(a,b)
    """

    def __init__(self,
                 alpha: float = 0.05,
                 beta: float = 0.05,
                 h0_dist: str = 'uniform',
                 h1_params: Tuple[float, float] = (2.0, 1.0)):
        """
        Args:
            alpha: Type I error rate (false positive)
            beta: Type II error rate (false negative)
            h0_dist: Null distribution ('uniform')
            h1_params: Alternative distribution params (Beta shape params)
        """
        self.alpha = alpha
        self.beta = beta
        self.h1_a, self.h1_b = h1_params

        # Wald boundaries
        self.A = np.log(beta / (1 - alpha))
        self.B = np.log((1 - beta) / alpha)

        # State
        self.log_likelihood_ratio = 0.0
        self.step = 0
        self.history = []

    def _log_likelihood_ratio(self, x: float) -> float:
        """
        Compute log-likelihood ratio for single observation.

        log(f_1(x) / f_0(x)) where:
        f_0(x) = 1 (uniform)
        f_1(x) = Beta(a,b) pdf
        """
        from scipy.stats import beta

        # Avoid log(0)
        x = np.clip(x, 1e-10, 1 - 1e-10)

        f1 = beta.pdf(x, self.h1_a, self.h1_b)
        f0 = 1.0  # Uniform density

        return np.log(f1 / f0)

    def reset(self):
        """Reset detector state."""
        self.log_likelihood_ratio = 0.0
        self.step = 0
        self.history = []

    def update(self, x: float) -> SequentialTestResult:
        """
        Process one observation.
        """
        self.step += 1

        llr = self._log_likelihood_ratio(x)
        self.log_likelihood_ratio += llr
        self.history.append(self.log_likelihood_ratio)

        # Decision
        if self.log_likelihood_ratio <= self.A:
            decision = DecisionState.ACCEPT_H0
        elif self.log_likelihood_ratio >= self.B:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=self.log_likelihood_ratio,
            decision=decision
        )

    def expected_sample_size(self, true_dist: str = 'h0') -> float:
        """
        Calculate expected sample size under given hypothesis.

        Wald's approximation:
        E[N | H_0] ≈ (1-α)A + αB) / E_0[log(f1/f0)]
        """
        from scipy.stats import beta

        if true_dist == 'h0':
            # Under H_0, E[log(f1/f0)] for uniform
            # Approximate via Monte Carlo
            samples = np.random.uniform(0, 1, 10000)
            mean_llr = np.mean([self._log_likelihood_ratio(x) for x in samples])

            if abs(mean_llr) < 1e-10:
                return np.inf

            return ((1 - self.alpha) * self.A + self.alpha * self.B) / mean_llr
        else:
            # Under H_1
            samples = beta.rvs(self.h1_a, self.h1_b, size=10000)
            mean_llr = np.mean([self._log_likelihood_ratio(x) for x in samples])

            return (self.beta * self.A + (1 - self.beta) * self.B) / mean_llr


class PageHinkleyDetector:
    """
    Page-Hinkley test for change detection.

    Similar to CUSUM but with different update rule.
    Better for detecting gradual changes.
    """

    def __init__(self, delta: float = 0.01,
                 threshold: float = 10.0,
                 alpha: float = 0.99):
        """
        Args:
            delta: Minimum detectable change
            threshold: Detection threshold
            alpha: Forgetting factor (0.99 = slow adaptation)
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha

        self.sum = 0.0
        self.mean = 0.0
        self.step = 0
        self.min_sum = 0.0
        self.history = []

    def update(self, x: float) -> SequentialTestResult:
        self.step += 1

        # Update running mean
        self.mean = self.alpha * self.mean + (1 - self.alpha) * x

        # Update sum
        self.sum += x - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)

        statistic = self.sum - self.min_sum
        self.history.append(statistic)

        if statistic > self.threshold:
            decision = DecisionState.REJECT_H0
        else:
            decision = DecisionState.CONTINUE

        return SequentialTestResult(
            step=self.step,
            statistic=statistic,
            decision=decision
        )
```

**Test Scenarios**:

| Test | Attack at step | Detection step | ARL₀ |
|------|---------------|----------------|------|
| CUSUM | 500 | ~520 | ~1000 |
| SPRT | 500 | ~510 | ~500 |
| Page-Hinkley | 500 | ~530 | ~800 |

**Validation Criteria**:
- [ ] Average Run Length under H₀ (ARL₀) > 500
- [ ] Detection delay < 50 samples after attack starts
- [ ] Comparison with TARA-M detection time
- [ ] Parameter sensitivity analysis

---

### Experiment 3.4: Bayesian Online Change Point Detection

**File**: `exp_3_4_bocpd.py`

**Concept**: BOCPD maintains a posterior distribution over the time since the last change point. Natural for streaming applications.

**Mathematical Foundation** (Adams & MacKay 2007):
```
P(r_t | x_{1:t}) ∝ Σ_{r_{t-1}} P(r_t | r_{t-1}) P(x_t | r_{t-1}, x_{1:t-1}^{(r)}) P(r_{t-1} | x_{1:t-1})

where:
- r_t is the run length at time t
- P(r_t | r_{t-1}) is the changepoint prior (often geometric)
- P(x_t | ...) is the predictive distribution (UPM)
```

**Implementation**:
```python
import numpy as np
from scipy import stats
from typing import List, Tuple

class BOCPDDetector:
    """
    Bayesian Online Changepoint Detection.

    Reference: Adams & MacKay (2007),
    "Bayesian Online Changepoint Detection"

    Maintains belief over run length (time since last change).
    Natural uncertainty quantification.
    """

    def __init__(self, hazard_rate: float = 1/200,
                 prior_alpha: float = 1.0,
                 prior_beta: float = 1.0):
        """
        Args:
            hazard_rate: Prior probability of changepoint (1/expected_run_length)
            prior_alpha, prior_beta: Beta prior for mean parameter
        """
        self.hazard = hazard_rate
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # State: run length distribution
        self.R = np.array([1.0])  # P(r=0) = 1 initially
        self.step = 0

        # Sufficient statistics for each run length
        self.alpha_params = [prior_alpha]
        self.beta_params = [prior_beta]
        self.n_obs = [0]

        # History for analysis
        self.run_length_history = []
        self.map_run_length_history = []
        self.changepoint_prob_history = []

    def _predictive_probability(self, x: float, alpha: float,
                                 beta: float) -> float:
        """
        Predictive probability under Beta-Bernoulli model.

        For continuous x in [0,1], use Beta predictive.
        """
        # Treat x as coming from Beta distribution
        # Predictive for new x given past
        return stats.beta.pdf(x, alpha, beta)

    def update(self, x: float) -> dict:
        """
        Process one observation.

        Returns:
            Dictionary with run length distribution and changepoint probability
        """
        self.step += 1

        # Compute predictive probabilities for each run length
        pred_probs = np.array([
            self._predictive_probability(x, a, b)
            for a, b in zip(self.alpha_params, self.beta_params)
        ])

        # Growth probabilities (no changepoint)
        growth_probs = self.R * pred_probs * (1 - self.hazard)

        # Changepoint probability (r_t = 0)
        cp_prob = np.sum(self.R * pred_probs) * self.hazard

        # New run length distribution
        new_R = np.concatenate([[cp_prob], growth_probs])
        new_R /= np.sum(new_R)  # Normalize

        # Update sufficient statistics
        # For r=0 (new segment), use prior
        new_alpha = [self.prior_alpha]
        new_beta = [self.prior_beta]
        new_n = [0]

        # For r>0, update with observation
        for a, b, n in zip(self.alpha_params, self.beta_params, self.n_obs):
            new_alpha.append(a + x)  # Treating x as success count proxy
            new_beta.append(b + (1 - x))
            new_n.append(n + 1)

        self.R = new_R
        self.alpha_params = new_alpha
        self.beta_params = new_beta
        self.n_obs = new_n

        # Compute quantities of interest
        map_run_length = np.argmax(self.R)
        changepoint_prob = self.R[0]
        expected_run_length = np.sum(np.arange(len(self.R)) * self.R)

        self.run_length_history.append(self.R.copy())
        self.map_run_length_history.append(map_run_length)
        self.changepoint_prob_history.append(changepoint_prob)

        return {
            'step': self.step,
            'map_run_length': map_run_length,
            'expected_run_length': expected_run_length,
            'changepoint_prob': changepoint_prob,
            'detected': changepoint_prob > 0.5,  # Simple threshold
            'run_length_distribution': self.R
        }

    def get_changepoints(self, threshold: float = 0.5) -> List[int]:
        """Get list of detected changepoint times."""
        return [i for i, p in enumerate(self.changepoint_prob_history)
                if p > threshold]

    def plot_run_length(self):
        """Visualization of run length posterior over time."""
        import matplotlib.pyplot as plt

        # Create run length matrix
        max_r = max(len(r) for r in self.run_length_history)
        R_matrix = np.zeros((len(self.run_length_history), max_r))

        for t, r in enumerate(self.run_length_history):
            R_matrix[t, :len(r)] = r

        plt.figure(figsize=(12, 6))
        plt.imshow(R_matrix.T, aspect='auto', origin='lower',
                   extent=[0, len(self.run_length_history), 0, max_r])
        plt.colorbar(label='P(run length)')
        plt.xlabel('Time')
        plt.ylabel('Run length')
        plt.title('BOCPD Run Length Posterior')
        return plt.gcf()
```

**Test Scenarios**:

| Scenario | Changepoint | Detection delay | False alarms |
|----------|-------------|-----------------|--------------|
| Single CP at t=500 | Yes | ~10-20 | 0-1 |
| Gradual drift | No | N/A | Should not detect |
| Multiple CPs | Yes × 3 | ~15 each | 0-2 |
| No change | No | N/A | < 5% rate |

**Validation Criteria**:
- [ ] Detects abrupt changes within 20 samples
- [ ] Posterior run length concentrates correctly
- [ ] Handles multiple changepoints
- [ ] Uncertainty quantification is calibrated

---

### Experiment 3.5: Classical Distribution Tests

**File**: `exp_3_5_classical_tests.py`

**Tests to implement**:

1. **Anderson-Darling** - More sensitive to tails than KS
2. **Cramér-von Mises** - Integrated squared difference of CDFs
3. **Chi-squared goodness-of-fit** - Binned comparison
4. **Lilliefors** - KS variant for unknown parameters

**Implementation**:
```python
import numpy as np
from scipy import stats
from typing import Dict, Any

class ClassicalTestSuite:
    """
    Suite of classical statistical tests for p-value uniformity.

    Under H_0 (no attack): p-values should be U[0,1]
    Under H_1 (attack): p-values deviate from uniformity
    """

    @staticmethod
    def anderson_darling(pvalues: np.ndarray) -> Dict[str, Any]:
        """
        Anderson-Darling test for uniformity.

        More sensitive to tail deviations than KS.

        A² = -n - (1/n) Σ (2i-1)[ln(F(X_i)) + ln(1-F(X_{n+1-i}))]
        """
        n = len(pvalues)
        sorted_p = np.sort(pvalues)

        # Avoid log(0)
        sorted_p = np.clip(sorted_p, 1e-10, 1 - 1e-10)

        i = np.arange(1, n + 1)
        A2 = -n - np.mean((2*i - 1) * (np.log(sorted_p) +
                                        np.log(1 - sorted_p[::-1])))

        # Critical values for uniform distribution
        # Approximate p-value using Marsaglia-Marsaglia formula
        if A2 < 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14*A2 - 223.73*A2**2)
        elif A2 < 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796*A2 - 59.938*A2**2)
        elif A2 < 0.6:
            p_value = np.exp(0.9177 - 4.279*A2 - 1.38*A2**2)
        else:
            p_value = np.exp(1.2937 - 5.709*A2 + 0.0186*A2**2)

        return {
            'test': 'Anderson-Darling',
            'statistic': A2,
            'p_value': max(0, min(1, p_value)),
            'critical_5pct': 2.492,
            'detected': A2 > 2.492
        }

    @staticmethod
    def cramer_von_mises(pvalues: np.ndarray) -> Dict[str, Any]:
        """
        Cramér-von Mises test for uniformity.

        W² = (1/12n) + Σ (F(X_i) - (2i-1)/(2n))²
        """
        n = len(pvalues)
        sorted_p = np.sort(pvalues)

        i = np.arange(1, n + 1)
        W2 = 1/(12*n) + np.sum((sorted_p - (2*i - 1)/(2*n))**2)

        # Transform for p-value (approximate)
        W2_star = W2 * (1 + 0.5/n)

        if W2_star < 0.0275:
            p_value = 1.0
        elif W2_star < 0.051:
            p_value = 0.5
        elif W2_star < 0.092:
            p_value = 0.25
        elif W2_star < 0.461:
            p_value = 0.05
        else:
            p_value = 0.01

        return {
            'test': 'Cramér-von Mises',
            'statistic': W2,
            'p_value': p_value,
            'critical_5pct': 0.461,
            'detected': W2 > 0.461
        }

    @staticmethod
    def chi_squared_uniformity(pvalues: np.ndarray,
                               n_bins: int = 10) -> Dict[str, Any]:
        """
        Chi-squared test for uniformity.

        Bins p-values and compares to expected uniform counts.
        """
        n = len(pvalues)
        expected = n / n_bins

        observed, _ = np.histogram(pvalues, bins=n_bins, range=(0, 1))

        chi2 = np.sum((observed - expected)**2 / expected)
        df = n_bins - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)

        return {
            'test': 'Chi-squared',
            'statistic': chi2,
            'df': df,
            'p_value': p_value,
            'critical_5pct': stats.chi2.ppf(0.95, df),
            'detected': p_value < 0.05
        }

    @staticmethod
    def kolmogorov_smirnov(pvalues: np.ndarray) -> Dict[str, Any]:
        """
        Kolmogorov-Smirnov test for uniformity.
        """
        stat, p_value = stats.kstest(pvalues, 'uniform')

        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': stat,
            'p_value': p_value,
            'detected': p_value < 0.05
        }

    @staticmethod
    def runs_test(pvalues: np.ndarray) -> Dict[str, Any]:
        """
        Runs test for randomness.

        Counts runs above/below median.
        """
        median = np.median(pvalues)
        signs = (pvalues > median).astype(int)

        # Count runs
        runs = 1 + np.sum(signs[1:] != signs[:-1])

        # Expected runs under H_0
        n_plus = np.sum(signs)
        n_minus = len(signs) - n_plus

        expected_runs = 1 + 2*n_plus*n_minus / len(signs)
        var_runs = (2*n_plus*n_minus * (2*n_plus*n_minus - len(signs))) / \
                   (len(signs)**2 * (len(signs) - 1))

        z = (runs - expected_runs) / np.sqrt(var_runs + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            'test': 'Runs',
            'statistic': runs,
            'z_score': z,
            'p_value': p_value,
            'detected': p_value < 0.05
        }

    def run_all(self, pvalues: np.ndarray) -> Dict[str, Dict]:
        """Run all tests and return combined results."""
        return {
            'anderson_darling': self.anderson_darling(pvalues),
            'cramer_von_mises': self.cramer_von_mises(pvalues),
            'chi_squared': self.chi_squared_uniformity(pvalues),
            'kolmogorov_smirnov': self.kolmogorov_smirnov(pvalues),
            'runs': self.runs_test(pvalues)
        }

    def combined_decision(self, pvalues: np.ndarray,
                          voting: str = 'majority') -> Dict[str, Any]:
        """
        Combined decision from multiple tests.

        Args:
            pvalues: Test p-values
            voting: 'majority', 'any', or 'all'
        """
        results = self.run_all(pvalues)

        detections = [r['detected'] for r in results.values()]

        if voting == 'majority':
            combined = sum(detections) > len(detections) / 2
        elif voting == 'any':
            combined = any(detections)
        else:  # all
            combined = all(detections)

        return {
            'individual_results': results,
            'n_detected': sum(detections),
            'n_tests': len(detections),
            'combined_detected': combined,
            'voting_method': voting
        }
```

**Comparison Table**:

| Test | Sensitivity to | Best for |
|------|---------------|----------|
| KS | Location/scale shift | General |
| Anderson-Darling | Tail behavior | Heavy-tailed attacks |
| Cramér-von Mises | Overall shape | Mixture attacks |
| Chi-squared | Discrete binning | Large samples |
| Runs | Serial correlation | Sequential attacks |

---

### Experiment 3.6: Ensemble Detection

**File**: `exp_3_6_ensemble_detection.py`

**Concept**: Combine multiple detectors for robustness. Different tests are sensitive to different types of attacks.

**Implementation**:
```python
import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    """Configuration for a single detector."""
    name: str
    detector_class: type
    params: Dict[str, Any]
    weight: float = 1.0

class EnsembleDetector:
    """
    Ensemble of multiple anomaly detectors.

    Combines decisions using:
    - Weighted voting
    - Soft voting (probability averaging)
    - Stacking (meta-learner)
    """

    def __init__(self, configs: List[DetectorConfig],
                 combination: str = 'weighted_vote'):
        """
        Args:
            configs: List of detector configurations
            combination: 'weighted_vote', 'soft_vote', or 'any'
        """
        self.configs = configs
        self.combination = combination

        # Instantiate detectors
        self.detectors = []
        for config in configs:
            detector = config.detector_class(**config.params)
            self.detectors.append({
                'name': config.name,
                'detector': detector,
                'weight': config.weight
            })

    def calibrate(self, calibration_data: np.ndarray):
        """Calibrate all detectors on reference data."""
        for d in self.detectors:
            if hasattr(d['detector'], 'calibrate'):
                d['detector'].calibrate(calibration_data)

    def test(self, test_data: np.ndarray) -> Dict[str, Any]:
        """
        Run all detectors and combine results.
        """
        results = []

        for d in self.detectors:
            try:
                if hasattr(d['detector'], 'test'):
                    result = d['detector'].test(test_data)
                else:
                    # Assume it's a function
                    result = d['detector'](test_data)

                results.append({
                    'name': d['name'],
                    'weight': d['weight'],
                    'detected': result.get('detected', False),
                    'confidence': result.get('confidence',
                                            result.get('p_value', 0.5)),
                    'full_result': result
                })
            except Exception as e:
                results.append({
                    'name': d['name'],
                    'weight': d['weight'],
                    'detected': False,
                    'confidence': 0.5,
                    'error': str(e)
                })

        # Combine results
        if self.combination == 'weighted_vote':
            weighted_sum = sum(r['weight'] * r['detected'] for r in results)
            total_weight = sum(r['weight'] for r in results)
            combined = weighted_sum > total_weight / 2
            combined_score = weighted_sum / total_weight

        elif self.combination == 'soft_vote':
            weighted_conf = sum(r['weight'] * r['confidence'] for r in results)
            total_weight = sum(r['weight'] for r in results)
            combined_score = weighted_conf / total_weight
            combined = combined_score > 0.5

        elif self.combination == 'any':
            combined = any(r['detected'] for r in results)
            combined_score = max(r['confidence'] for r in results)

        else:  # all
            combined = all(r['detected'] for r in results)
            combined_score = min(r['confidence'] for r in results)

        return {
            'individual_results': results,
            'combined_detected': combined,
            'combined_score': combined_score,
            'n_detected': sum(r['detected'] for r in results),
            'n_detectors': len(results)
        }

    def optimize_weights(self, labeled_data: List[tuple],
                         metric: str = 'f1') -> np.ndarray:
        """
        Optimize detector weights using labeled data.

        Args:
            labeled_data: List of (test_data, is_attack) tuples
            metric: Optimization metric ('f1', 'accuracy', 'auc')
        """
        from scipy.optimize import minimize

        def objective(weights):
            # Normalize weights
            weights = np.abs(weights) / np.sum(np.abs(weights))

            # Update weights
            for i, d in enumerate(self.detectors):
                d['weight'] = weights[i]

            # Evaluate on labeled data
            tp, fp, tn, fn = 0, 0, 0, 0
            for data, label in labeled_data:
                result = self.test(data)
                pred = result['combined_detected']

                if label and pred:
                    tp += 1
                elif not label and pred:
                    fp += 1
                elif not label and not pred:
                    tn += 1
                else:
                    fn += 1

            if metric == 'f1':
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                return -f1  # Minimize negative F1
            elif metric == 'accuracy':
                return -(tp + tn) / len(labeled_data)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Initial weights
        x0 = np.ones(len(self.detectors)) / len(self.detectors)

        result = minimize(objective, x0, method='Nelder-Mead')
        optimal_weights = np.abs(result.x) / np.sum(np.abs(result.x))

        # Update final weights
        for i, d in enumerate(self.detectors):
            d['weight'] = optimal_weights[i]

        return optimal_weights
```

---

## Category B: Finite-Key Security Analysis

### Experiment 3.7: Finite-Key Bounds Implementation

**File**: `exp_3_7_finite_key.py`

**Concept**: Real QKD sessions have finite samples. Security proofs must account for statistical fluctuations.

**Key formulas** (Tomamichel et al. 2012):

```
Asymptotic key rate:
r_∞ = 1 - h(QBER) - leak_EC

Finite-key rate:
r_finite = r_∞ - Δ_AEP - Δ_EC - Δ_PA

where:
Δ_AEP ≈ 4 log(2/ε_s) × √(log(2/ε_PE) / n)   (Asymptotic Equipartition)
Δ_EC ≈ log(1/ε_EC) / n                        (Error correction)
Δ_PA ≈ 2 log(1/ε_PA) / n                      (Privacy amplification)
```

**Implementation**:
```python
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class FiniteKeyParameters:
    """Security parameters for finite-key analysis."""
    epsilon_sec: float = 1e-10      # Total security parameter
    epsilon_cor: float = 1e-15      # Correctness parameter
    epsilon_pe: float = 1e-10       # Parameter estimation error
    epsilon_ec: float = 1e-10       # Error correction failure
    epsilon_pa: float = 1e-10       # Privacy amplification error

    def validate(self):
        """Check that parameters are consistent."""
        total = self.epsilon_pe + self.epsilon_ec + self.epsilon_pa
        if total > self.epsilon_sec:
            raise ValueError(f"Sum of epsilons ({total}) exceeds total ({self.epsilon_sec})")

def binary_entropy(p: float) -> float:
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

class FiniteKeyAnalyzer:
    """
    Finite-key security analysis for QKD.

    References:
    - Tomamichel et al. (2012) "Tight finite-key analysis for QKD"
    - Lim et al. (2014) "Concise security bounds for practical decoy-state QKD"
    """

    def __init__(self, params: FiniteKeyParameters = None):
        self.params = params or FiniteKeyParameters()
        self.params.validate()

    def qber_confidence_bound(self, n_sample: int, n_errors: int,
                               confidence: float = None) -> Tuple[float, float]:
        """
        Compute confidence interval for QBER.

        Uses Clopper-Pearson exact method.

        Returns:
            (qber_estimate, qber_upper_bound)
        """
        from scipy.stats import beta

        if confidence is None:
            confidence = 1 - self.params.epsilon_pe

        alpha = 1 - confidence
        qber = n_errors / n_sample

        # Upper bound (one-sided)
        if n_errors == n_sample:
            upper = 1.0
        else:
            upper = beta.ppf(1 - alpha, n_errors + 1, n_sample - n_errors)

        return qber, upper

    def delta_aep(self, n: int) -> float:
        """
        Asymptotic Equipartition Property correction.

        Accounts for finite-block-length effect on entropy estimation.
        """
        eps_s = self.params.epsilon_sec
        eps_pe = self.params.epsilon_pe

        # Simplified formula from Tomamichel
        return 4 * np.log2(2 / eps_s) * np.sqrt(np.log2(2 / eps_pe) / n)

    def delta_ec(self, n: int, qber: float, f_ec: float = 1.16) -> float:
        """
        Error correction overhead.

        Args:
            n: Number of bits
            qber: Quantum bit error rate
            f_ec: Error correction efficiency (CASCADE ≈ 1.16)
        """
        # Bits leaked during EC
        if qber <= 0:
            leak = 0
        else:
            leak = f_ec * n * binary_entropy(qber)

        # Failure probability contribution
        failure_term = np.log2(1 / self.params.epsilon_ec)

        return leak / n + failure_term / n

    def delta_pa(self, n: int) -> float:
        """
        Privacy amplification correction.

        Accounts for hash function security.
        """
        return 2 * np.log2(1 / self.params.epsilon_pa) / n

    def compute_key_rate(self, n_sifted: int, n_sample: int, n_errors: int,
                          chsh_value: float = None) -> Dict:
        """
        Compute finite-key rate.

        Args:
            n_sifted: Total sifted bits
            n_sample: Bits used for QBER estimation
            n_errors: Errors found in sample
            chsh_value: CHSH value (for DI analysis)

        Returns:
            Dictionary with rate and all intermediate values
        """
        n_key = n_sifted - n_sample  # Bits available for key

        # QBER estimation
        qber_est, qber_upper = self.qber_confidence_bound(n_sample, n_errors)

        # Check security threshold
        if qber_upper >= 0.11:
            return {
                'secure': False,
                'reason': f'QBER upper bound ({qber_upper:.3f}) exceeds threshold',
                'qber_estimate': qber_est,
                'qber_upper': qber_upper,
                'key_rate': 0,
                'key_length': 0
            }

        # Asymptotic rate
        r_asymptotic = 1 - binary_entropy(qber_upper)

        # Finite-key corrections
        d_aep = self.delta_aep(n_key)
        d_ec = self.delta_ec(n_key, qber_upper)
        d_pa = self.delta_pa(n_key)

        # Total correction
        total_correction = d_aep + d_ec + d_pa

        # Finite-key rate
        r_finite = max(0, r_asymptotic - total_correction)

        # Final key length
        key_length = int(n_key * r_finite)

        return {
            'secure': key_length > 0,
            'qber_estimate': qber_est,
            'qber_upper': qber_upper,
            'n_sifted': n_sifted,
            'n_sample': n_sample,
            'n_key_bits': n_key,
            'r_asymptotic': r_asymptotic,
            'delta_aep': d_aep,
            'delta_ec': d_ec,
            'delta_pa': d_pa,
            'total_correction': total_correction,
            'r_finite': r_finite,
            'key_length': key_length,
            'key_rate_per_pair': key_length / n_sifted if n_sifted > 0 else 0,
            'security_parameter': self.params.epsilon_sec
        }

    def minimum_samples_for_key(self, target_key_bits: int,
                                 expected_qber: float = 0.03) -> int:
        """
        Calculate minimum sifted bits needed for target key length.

        Binary search for required n.
        """
        # Start with optimistic estimate
        n_low = target_key_bits
        n_high = target_key_bits * 100

        while n_high - n_low > 100:
            n_mid = (n_low + n_high) // 2

            # Estimate sample size (15% of sifted)
            n_sample = int(0.15 * n_mid)
            n_errors = int(expected_qber * n_sample)

            result = self.compute_key_rate(n_mid, n_sample, n_errors)

            if result['key_length'] >= target_key_bits:
                n_high = n_mid
            else:
                n_low = n_mid

        return n_high

    def key_rate_vs_distance(self, distances_km: np.ndarray,
                              n_pairs_per_km: int = 10000,
                              fiber_loss_db_per_km: float = 0.2,
                              detector_efficiency: float = 0.1,
                              dark_count_rate: float = 1e-6) -> Dict:
        """
        Calculate key rate as function of distance.

        Includes:
        - Fiber attenuation
        - Detector dark counts
        - Finite-key effects
        """
        results = []

        for d in distances_km:
            # Transmission probability
            eta = 10 ** (-fiber_loss_db_per_km * d / 10) * detector_efficiency

            # Effective number of detected pairs
            n_detected = int(n_pairs_per_km * eta)

            # QBER contribution from dark counts
            signal_rate = eta
            noise_rate = dark_count_rate
            qber_intrinsic = 0.01  # Base QBER
            qber_dark = noise_rate / (signal_rate + noise_rate + 1e-10)
            qber_total = qber_intrinsic + qber_dark

            if n_detected < 1000:
                results.append({
                    'distance_km': d,
                    'transmission': eta,
                    'n_detected': n_detected,
                    'qber': qber_total,
                    'key_rate': 0,
                    'secure': False,
                    'reason': 'Insufficient detections'
                })
                continue

            # Sifting
            n_sifted = int(n_detected * 0.5)
            n_sample = int(0.15 * n_sifted)
            n_errors = int(qber_total * n_sample)

            result = self.compute_key_rate(n_sifted, n_sample, n_errors)
            result['distance_km'] = d
            result['transmission'] = eta
            result['n_detected'] = n_detected

            results.append(result)

        return {
            'distances': distances_km.tolist(),
            'results': results,
            'max_secure_distance': max([r['distance_km'] for r in results
                                        if r.get('secure', False)], default=0)
        }
```

**Test Scenarios**:

| n_sifted | QBER | Asymptotic rate | Finite rate | Key bits |
|----------|------|-----------------|-------------|----------|
| 1,000 | 3% | 85% | 35% | 350 |
| 10,000 | 3% | 85% | 75% | 7,500 |
| 100,000 | 3% | 85% | 82% | 82,000 |
| 10,000 | 8% | 42% | 25% | 2,500 |

---

## Category C: Channel Effects & Distance

### Experiment 3.8: Fiber Channel Model

**File**: `exp_3_8_fiber_channel.py`

**Effects to model**:

1. **Attenuation**: `P(L) = P_0 × 10^(-αL/10)`, α ≈ 0.2 dB/km
2. **Chromatic dispersion**: Pulse broadening, timing jitter
3. **Polarization mode dispersion**: Random polarization rotation
4. **Backscatter**: Rayleigh scattering noise

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class FiberParameters:
    """Standard single-mode fiber parameters."""
    attenuation_db_km: float = 0.2      # dB/km at 1550nm
    dispersion_ps_nm_km: float = 17.0   # ps/(nm·km)
    pmd_ps_sqrt_km: float = 0.1         # ps/√km
    rayleigh_coefficient: float = 1e-7  # Backscatter coefficient
    nonlinear_coefficient: float = 1.3  # W^-1 km^-1 (for high power)

@dataclass
class DetectorParameters:
    """Single-photon detector parameters."""
    efficiency: float = 0.1             # Detection efficiency
    dark_count_rate: float = 1e-6       # Dark counts per gate
    dead_time_ns: float = 50            # Dead time
    timing_jitter_ps: float = 100       # Timing jitter

class FiberChannelSimulator:
    """
    Realistic fiber quantum channel simulator.

    Models all major impairments for QKD over optical fiber.
    """

    def __init__(self, fiber: FiberParameters = None,
                 detector: DetectorParameters = None):
        self.fiber = fiber or FiberParameters()
        self.detector = detector or DetectorParameters()

    def transmission_probability(self, distance_km: float) -> float:
        """
        Calculate end-to-end transmission probability.

        Includes fiber loss and detector efficiency.
        """
        fiber_transmission = 10 ** (-self.fiber.attenuation_db_km * distance_km / 10)
        return fiber_transmission * self.detector.efficiency

    def dark_count_contribution(self, distance_km: float,
                                 gate_width_ns: float = 1.0) -> float:
        """
        Calculate QBER contribution from dark counts.

        Dark counts add random noise to both 0 and 1 outcomes.
        """
        eta = self.transmission_probability(distance_km)

        # Signal detection rate
        p_signal = eta

        # Noise detection rate (dark counts + backscatter)
        p_noise = self.detector.dark_count_rate * gate_width_ns * 1e-9

        # QBER from dark counts (appear as random bits)
        if p_signal + p_noise < 1e-15:
            return 0.5  # No signal = maximum QBER

        qber_dark = 0.5 * p_noise / (p_signal + p_noise)
        return qber_dark

    def timing_jitter_effect(self, distance_km: float,
                              laser_linewidth_ghz: float = 0.001) -> float:
        """
        Calculate timing jitter from dispersion.

        Returns standard deviation of arrival time in ps.
        """
        # Chromatic dispersion contribution
        wavelength_spread_nm = laser_linewidth_ghz * 0.008  # approx conversion
        dispersion_jitter = self.fiber.dispersion_ps_nm_km * distance_km * wavelength_spread_nm

        # PMD contribution (random walk)
        pmd_jitter = self.fiber.pmd_ps_sqrt_km * np.sqrt(distance_km)

        # Detector jitter
        detector_jitter = self.detector.timing_jitter_ps

        # Total (quadrature sum)
        total_jitter = np.sqrt(dispersion_jitter**2 + pmd_jitter**2 + detector_jitter**2)

        return total_jitter

    def polarization_drift(self, distance_km: float,
                           time_hours: float = 1.0) -> float:
        """
        Model polarization state drift over fiber.

        Returns expected polarization error rate.
        """
        # Simplified model: random walk in polarization
        # More drift with longer fiber and time
        drift_rate = 0.001 * np.sqrt(distance_km * time_hours)
        return min(drift_rate, 0.5)

    def simulate_qkd_over_fiber(self, distance_km: float,
                                 n_pairs: int = 10000,
                                 intrinsic_qber: float = 0.01) -> Dict:
        """
        Full simulation of QKD over fiber channel.

        Returns expected performance metrics.
        """
        # Transmission
        eta = self.transmission_probability(distance_km)
        n_detected = int(n_pairs * eta)

        # QBER contributions
        qber_intrinsic = intrinsic_qber
        qber_dark = self.dark_count_contribution(distance_km)
        qber_polarization = self.polarization_drift(distance_km)

        # Total QBER (approximately additive for small values)
        qber_total = qber_intrinsic + qber_dark + qber_polarization
        qber_total = min(qber_total, 0.5)

        # Timing jitter
        jitter_ps = self.timing_jitter_effect(distance_km)

        # Coincidence window (must be > jitter for good detection)
        coincidence_window_ps = max(500, 3 * jitter_ps)

        # Accidental coincidences
        accidental_rate = (self.detector.dark_count_rate *
                          coincidence_window_ps * 1e-12)

        return {
            'distance_km': distance_km,
            'n_pairs_sent': n_pairs,
            'transmission': eta,
            'n_detected': n_detected,
            'qber_intrinsic': qber_intrinsic,
            'qber_dark_counts': qber_dark,
            'qber_polarization': qber_polarization,
            'qber_total': qber_total,
            'timing_jitter_ps': jitter_ps,
            'coincidence_window_ps': coincidence_window_ps,
            'accidental_rate': accidental_rate,
            'secure': qber_total < 0.11 and n_detected > 1000
        }

    def distance_sweep(self, max_distance_km: float = 200,
                       n_points: int = 50,
                       n_pairs: int = 100000) -> Dict:
        """
        Sweep distance and calculate key rate.
        """
        distances = np.linspace(1, max_distance_km, n_points)
        results = []

        for d in distances:
            sim = self.simulate_qkd_over_fiber(d, n_pairs)

            if sim['secure']:
                # Estimate key rate
                n_sifted = int(sim['n_detected'] * 0.5)
                key_rate = max(0, 1 - 2 * binary_entropy(sim['qber_total']))
                key_bits = int(n_sifted * key_rate * 0.8)  # 80% for finite-key
            else:
                key_rate = 0
                key_bits = 0

            sim['key_rate'] = key_rate
            sim['key_bits'] = key_bits
            results.append(sim)

        # Find maximum secure distance
        secure_distances = [r['distance_km'] for r in results if r['secure']]
        max_secure = max(secure_distances) if secure_distances else 0

        return {
            'distances': distances.tolist(),
            'results': results,
            'max_secure_distance_km': max_secure,
            'fiber_params': self.fiber.__dict__,
            'detector_params': self.detector.__dict__
        }
```

---

### Experiment 3.9: Free-Space Channel Model

**File**: `exp_3_9_free_space.py`

**Effects**:
- Geometric beam spreading
- Atmospheric turbulence (scintillation)
- Background light noise
- Pointing/tracking errors

*(Detailed implementation similar to fiber model)*

---

## Category D: Advanced Eve Attacks

### Experiment 3.10: Collective Attacks

**File**: `exp_3_10_collective_attacks.py`

**Concept**: Eve applies same attack to each qubit, stores quantum memory, measures optimally at end.

```python
class CollectiveAttackSimulator:
    """
    Simulates collective attacks where Eve:
    1. Interacts identically with each qubit
    2. Stores her ancilla in quantum memory
    3. Performs optimal collective measurement at end

    Security bound: Devetak-Winter rate still applies
    """

    def __init__(self):
        self.eve_info_cache = {}

    def optimal_cloning_attack(self, n_qubits: int,
                                fidelity: float = 5/6) -> Dict:
        """
        Universal symmetric cloning attack.

        Eve creates optimal clones of transmitted qubits.
        Fidelity 5/6 is optimal for 1→2 symmetric cloning.
        """
        # QBER introduced by cloning
        qber = 1 - fidelity

        # Eve's information per qubit (Holevo bound)
        eve_info = self._holevo_information(fidelity)

        # Alice-Bob mutual information
        alice_bob_info = 1 - binary_entropy(qber)

        # Secure key rate (Devetak-Winter)
        key_rate = max(0, alice_bob_info - eve_info)

        return {
            'attack_type': 'optimal_cloning',
            'fidelity': fidelity,
            'qber': qber,
            'eve_info_per_qubit': eve_info,
            'alice_bob_info': alice_bob_info,
            'key_rate': key_rate,
            'n_qubits': n_qubits,
            'expected_key_bits': int(n_qubits * 0.5 * key_rate)  # After sifting
        }

    def _holevo_information(self, fidelity: float) -> float:
        """
        Calculate Holevo information for clone.

        χ(Eve) = S(ρ_E) - Σ_x p(x) S(ρ_E^x)

        For symmetric cloning with fidelity F:
        χ ≈ h(F) where h is binary entropy
        """
        return binary_entropy(fidelity)

    def intercept_resend_collective(self, n_qubits: int,
                                     intercept_fraction: float) -> Dict:
        """
        Collective intercept-resend attack.

        Eve intercepts fraction p of qubits, measures in random basis,
        resends. Stores measurement results in classical memory.
        """
        p = intercept_fraction

        # QBER from interception
        qber = p * 0.25

        # Eve's information (she knows intercepted bits perfectly)
        eve_info = p * 0.5  # 50% in wrong basis are useless

        # Alice-Bob information
        alice_bob_info = 1 - binary_entropy(qber)

        key_rate = max(0, alice_bob_info - eve_info)

        return {
            'attack_type': 'intercept_resend_collective',
            'intercept_fraction': p,
            'qber': qber,
            'eve_info_per_qubit': eve_info,
            'alice_bob_info': alice_bob_info,
            'key_rate': key_rate,
            'detection_probability': 1 - (1 - qber)**100  # Prob of detecting in 100 samples
        }

    def beam_splitter_attack(self, n_qubits: int,
                              splitting_ratio: float) -> Dict:
        """
        Beam splitter attack (for weak coherent pulses).

        Eve splits off fraction η of each pulse.
        For multi-photon pulses, she can get full information
        without introducing errors.
        """
        eta = splitting_ratio

        # Assuming Poisson-distributed photon numbers with mean μ
        mu = 0.5  # Typical QKD mean photon number

        # Probability of multi-photon pulse
        p_multi = 1 - np.exp(-mu) - mu * np.exp(-mu)

        # Eve's information from multi-photon pulses
        eve_info_multi = p_multi * eta

        # Single-photon contribution (introduces errors)
        p_single = mu * np.exp(-mu)
        qber_single = eta * p_single * 0.25

        # Total
        qber = qber_single
        eve_info = eve_info_multi + qber * 2  # Simplified bound

        return {
            'attack_type': 'beam_splitter',
            'splitting_ratio': eta,
            'mean_photon_number': mu,
            'p_multi_photon': p_multi,
            'qber': qber,
            'eve_info_per_qubit': eve_info,
            'vulnerable_to_pns': p_multi > 0.1
        }
```

---

### Experiment 3.11: Coherent Attacks

**File**: `exp_3_11_coherent_attacks.py`

**Concept**: Most general attack - Eve entangles with all qubits jointly.

Security proven via entropy accumulation theorem.

---

### Experiment 3.12: Side-Channel Attacks (Simulated)

**File**: `exp_3_12_side_channel_sim.py`

**Attacks to simulate**:
- Detector blinding
- Time-shift attack
- Trojan horse attack
- Detector efficiency mismatch

---

## Category E: Machine Learning Detection

### Experiment 3.13: Autoencoder Anomaly Detection

**File**: `exp_3_13_autoencoder.py`

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class QKDAnomalyAutoencoder(nn.Module):
    """
    Autoencoder for QKD anomaly detection.

    Trained on honest (calibration) data.
    High reconstruction error indicates anomaly.
    """

    def __init__(self, input_dim: int = 10,
                 hidden_dims: list = [64, 32, 16],
                 latent_dim: int = 8):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # Output in [0,1]
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def reconstruction_error(self, x):
        reconstructed, _ = self.forward(x)
        return torch.mean((x - reconstructed)**2, dim=1)


class AutoencoderDetector:
    """
    Anomaly detector using trained autoencoder.
    """

    def __init__(self, input_dim: int = 10, device: str = 'mps'):
        self.device = device
        self.model = QKDAnomalyAutoencoder(input_dim=input_dim)
        self.model.to(device)
        self.threshold = None

    def prepare_features(self, qkd_data: dict) -> np.ndarray:
        """
        Extract features from QKD data.

        Features:
        - CHSH correlators (4)
        - QBER
        - Sifting rate
        - Correlation statistics
        - P-value moments
        """
        features = []

        # CHSH correlators
        if 'correlators' in qkd_data:
            features.extend(list(qkd_data['correlators'].values()))

        # Add other features
        features.append(qkd_data.get('qber', 0))
        features.append(qkd_data.get('sifting_rate', 0.5))
        features.append(qkd_data.get('chsh_value', 2.8) / 2.828)  # Normalized

        # P-value statistics if available
        if 'p_values' in qkd_data:
            pvals = np.array(qkd_data['p_values'])
            features.extend([
                np.mean(pvals),
                np.std(pvals),
                np.percentile(pvals, 25),
                np.percentile(pvals, 75)
            ])

        return np.array(features, dtype=np.float32)

    def train(self, calibration_data: list, epochs: int = 100,
              batch_size: int = 32, learning_rate: float = 1e-3):
        """
        Train autoencoder on calibration (honest) data.
        """
        # Prepare dataset
        features = np.array([self.prepare_features(d) for d in calibration_data])
        dataset = TensorDataset(torch.FloatTensor(features))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                reconstructed, _ = self.model(x)
                loss = criterion(reconstructed, x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

        # Set threshold based on calibration errors
        self.model.eval()
        with torch.no_grad():
            cal_features = torch.FloatTensor(features).to(self.device)
            errors = self.model.reconstruction_error(cal_features).cpu().numpy()
            self.threshold = np.percentile(errors, 95)  # 5% false positive rate

        return {'final_loss': total_loss/len(loader), 'threshold': self.threshold}

    def detect(self, test_data: dict) -> dict:
        """
        Detect anomaly in test data.
        """
        self.model.eval()
        features = self.prepare_features(test_data)

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            error = self.model.reconstruction_error(x).item()

        return {
            'reconstruction_error': error,
            'threshold': self.threshold,
            'anomaly_score': error / self.threshold,
            'detected': error > self.threshold
        }
```

---

### Experiment 3.14: LSTM Sequence Detector

**File**: `exp_3_14_lstm_detector.py`

**Concept**: Use LSTM to model temporal patterns in QKD data. Deviation from predicted pattern indicates attack.

---

### Experiment 3.15: Isolation Forest

**File**: `exp_3_15_isolation_forest.py`

**Concept**: Unsupervised anomaly detection based on isolation in random trees. Fast, no training labels needed.

---

### Experiment 3.16: One-Class SVM

**File**: `exp_3_16_one_class_svm.py`

**Concept**: Learn boundary around normal data in kernel space.

---

### Experiment 3.17: ML Ensemble

**File**: `exp_3_17_ml_ensemble.py`

**Concept**: Combine multiple ML models for robust detection.

---

## Category F: Protocol Optimizations

### Experiment 3.18: Adaptive QBER Sampling

**File**: `exp_3_18_adaptive_sampling.py`

```python
class AdaptiveQBERSampler:
    """
    Adaptive sampling strategy for QBER estimation.

    Start with high sampling rate for quick security check.
    Reduce rate as confidence increases.
    """

    def __init__(self, initial_rate: float = 0.25,
                 min_rate: float = 0.05,
                 confidence_target: float = 0.99):
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.confidence_target = confidence_target

        self.n_sampled = 0
        self.n_errors = 0
        self.current_rate = initial_rate

    def should_sample(self, bit_index: int) -> bool:
        """Decide whether to sample this bit."""
        return np.random.random() < self.current_rate

    def update(self, is_error: bool):
        """Update after sampling a bit."""
        self.n_sampled += 1
        if is_error:
            self.n_errors += 1

        # Update sampling rate based on confidence
        if self.n_sampled >= 100:
            qber_est = self.n_errors / self.n_sampled
            ci_width = 1.96 * np.sqrt(qber_est * (1-qber_est) / self.n_sampled)

            # If confidence interval is small enough, reduce sampling
            if ci_width < 0.01:  # 1% precision
                self.current_rate = max(self.min_rate, self.current_rate * 0.9)

    def get_estimate(self) -> dict:
        """Get current QBER estimate."""
        if self.n_sampled == 0:
            return {'qber': 0, 'ci_width': 1, 'n_sampled': 0}

        qber = self.n_errors / self.n_sampled
        ci_width = 1.96 * np.sqrt(qber * (1-qber) / self.n_sampled)

        return {
            'qber': qber,
            'ci_width': ci_width,
            'n_sampled': self.n_sampled,
            'current_rate': self.current_rate,
            'bits_saved': self._bits_saved()
        }

    def _bits_saved(self) -> int:
        """Estimate bits saved compared to fixed-rate sampling."""
        # Assuming fixed 15% rate
        fixed_samples = int(self.n_sampled / self.current_rate * 0.15)
        return max(0, fixed_samples - self.n_sampled)
```

---

### Experiment 3.19: LDPC Error Correction

**File**: `exp_3_19_ldpc.py`

**Concept**: Low-Density Parity-Check codes are more efficient than CASCADE (f ≈ 1.05 vs 1.16).

---

### Experiment 3.20: Polar Codes

**File**: `exp_3_20_polar_codes.py`

**Concept**: Capacity-achieving codes with efficient encoding/decoding.

---

### Experiment 3.21: Continuous-Variable Features

**File**: `exp_3_21_cv_features.py`

**Concept**: Extract additional information from continuous measurement outcomes (not just binary).

---

## Experiment Index & Dependencies

### Dependency Graph

```
3.1 TARA-W ──────────────────────┐
3.2 TARA-MMD ────────────────────┤
3.3 Sequential (CUSUM/SPRT) ─────┼──► 3.6 Ensemble ──► 3.17 ML Ensemble
3.4 BOCPD ───────────────────────┤
3.5 Classical Tests ─────────────┘

3.7 Finite-Key ──────────────────┬──► 3.8 Fiber Channel ──► Distance Analysis
                                 └──► 3.9 Free-Space

3.10 Collective Attacks ─────────┐
3.11 Coherent Attacks ───────────┼──► Attack Detection Validation
3.12 Side-Channel (Sim) ─────────┘

3.13 Autoencoder ────────────────┐
3.14 LSTM ───────────────────────┤
3.15 Isolation Forest ───────────┼──► 3.17 ML Ensemble
3.16 One-Class SVM ──────────────┘

3.18 Adaptive Sampling ──────────┐
3.19 LDPC ───────────────────────┼──► Optimized Pipeline
3.20 Polar Codes ────────────────┘
```

### Implementation Order (Recommended)

**Week 1: Statistical Tests**
- Day 1-2: Exp 3.1 (Wasserstein), 3.2 (MMD)
- Day 3-4: Exp 3.3 (Sequential), 3.4 (BOCPD)
- Day 5: Exp 3.5 (Classical), 3.6 (Ensemble)

**Week 2: Security Analysis**
- Day 1-2: Exp 3.7 (Finite-Key)
- Day 3-4: Exp 3.8 (Fiber), 3.9 (Free-Space)
- Day 5: Exp 3.10-3.12 (Attacks)

**Week 3: Machine Learning**
- Day 1-2: Exp 3.13 (Autoencoder), 3.14 (LSTM)
- Day 3: Exp 3.15 (Isolation Forest), 3.16 (SVM)
- Day 4-5: Exp 3.17 (Ensemble), comparison

**Week 4: Optimizations**
- Day 1-2: Exp 3.18 (Adaptive Sampling)
- Day 3-4: Exp 3.19-3.20 (Alternative EC)
- Day 5: Integration & Final Testing

---

## Success Criteria

### Per-Experiment Validation

| Experiment | Primary Metric | Target |
|------------|---------------|--------|
| 3.1-3.6 | Detection rate | >90% for strong attacks |
| 3.7 | Key rate accuracy | Within 5% of theory |
| 3.8-3.9 | Distance prediction | Match literature |
| 3.10-3.12 | Security bound | Conservative |
| 3.13-3.17 | AUC-ROC | >0.95 |
| 3.18-3.20 | Efficiency gain | >10% over baseline |

### Overall Phase 3 Success

- [ ] At least 3 statistical tests outperform TARA-K
- [ ] Finite-key analysis matches Tomamichel et al.
- [ ] ML detection achieves >95% AUC
- [ ] Maximum secure fiber distance calculated
- [ ] All attacks properly bounded
- [ ] At least one EC method more efficient than CASCADE

---

## File Structure

```
Kamran_Experiments/
├── phase3_advanced/
│   ├── statistical_tests/
│   │   ├── exp_3_1_tara_wasserstein.py
│   │   ├── exp_3_2_tara_mmd.py
│   │   ├── exp_3_3_sequential_tests.py
│   │   ├── exp_3_4_bocpd.py
│   │   ├── exp_3_5_classical_tests.py
│   │   └── exp_3_6_ensemble_detection.py
│   │
│   ├── security_analysis/
│   │   ├── exp_3_7_finite_key.py
│   │   ├── exp_3_8_fiber_channel.py
│   │   ├── exp_3_9_free_space.py
│   │   ├── exp_3_10_collective_attacks.py
│   │   ├── exp_3_11_coherent_attacks.py
│   │   └── exp_3_12_side_channel_sim.py
│   │
│   ├── ml_detection/
│   │   ├── exp_3_13_autoencoder.py
│   │   ├── exp_3_14_lstm_detector.py
│   │   ├── exp_3_15_isolation_forest.py
│   │   ├── exp_3_16_one_class_svm.py
│   │   └── exp_3_17_ml_ensemble.py
│   │
│   ├── optimizations/
│   │   ├── exp_3_18_adaptive_sampling.py
│   │   ├── exp_3_19_ldpc.py
│   │   ├── exp_3_20_polar_codes.py
│   │   └── exp_3_21_cv_features.py
│   │
│   └── utils/
│       ├── statistical_utils.py
│       ├── ml_utils.py
│       ├── channel_models.py
│       └── attack_models.py
│
├── results/
│   └── phase3/
│
└── phase4_hardware/
    ├── exp_4_1_ibm_validation.py
    └── exp_4_2_ionq_validation.py
```

---

## References

### Statistical Methods
- Gretton et al. (2012) - MMD two-sample test
- Adams & MacKay (2007) - BOCPD
- Page (1954) - CUSUM
- Wald (1945) - SPRT

### Finite-Key Security
- Tomamichel et al. (2012) - Tight finite-key analysis
- Lim et al. (2014) - Practical decoy-state bounds
- Renner (2008) - Security of QKD

### Machine Learning
- Goodfellow et al. (2016) - Deep Learning
- Liu et al. (2008) - Isolation Forest
- Schölkopf et al. (2001) - One-class SVM

### Channel Models
- Gisin et al. (2002) - Quantum cryptography review
- Takeoka et al. (2014) - Fundamental rate-loss tradeoff

---

*Phase 3 Planning Complete*
*Total: 21 experiments, ~4 weeks development*
*Next: Implementation begins after approval*
