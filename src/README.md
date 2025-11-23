# Source Code Documentation

This directory contains the core implementation of the Quantum Conformal Entanglement framework. Below is a detailed description of every file.

## Root
- **`cli.py`**: The main command-line interface. Entry point for running all experiments defined in `configs/`.
- **`__init__.py`**: Package initialization.

## `analysis/` (Data Analysis & Validation)
- **`adversarial.py`**: Implements the adversarial LHV optimization loop to test ML witness robustness.
- **`bootstrap_analysis.py`**: Performs bootstrap resampling to estimate confidence intervals and statistical power.
- **`cew_bounds.py`**: Verifies the theoretical $(1+\alpha)^2$ bounds for CEW.
- **`cew_calibration.py`**: Analyzes the calibration of Conformal Entanglement Witnesses.
- **`ml_explain.py`**: Uses SHAP and feature importance metrics to explain the ML witness models.
- **`__init__.py`**: Module initialization.

## `cp/` (Conformal Prediction Core)
- **`mondrian.py`**: Implements Mondrian conformal prediction (stratified by context).
- **`scores.py`**: Defines nonconformity score functions (e.g., Bell-based, Polynomial, KNN).
- **`selective.py`**: Implements selective conformal prediction (rejecting uncertain predictions).
- **`split.py`**: Core split-conformal prediction logic (Train/Calibrate/Test splitting).
- **`__init__.py`**: Module initialization.

## `io/` (Input/Output & Hardware)
- **`config.py`**: Utilities for parsing YAML/JSON configuration files.
- **`external_data.py`**: Parsers for external datasets (e.g., NIST, Delft).
- **`notes.py`**: Helper for logging experimental notes.
- **`paths.py`**: Defines absolute paths for project directories.
- **`run_ibm_real.py`**: Interface for executing circuits on IBM Quantum hardware via Qiskit Runtime.
- **`__init__.py`**: Module initialization.

## `sims/` (Simulations)
- **`chsh.py`**: Simulates the CHSH experiment using quantum mechanics (Bell states).
- **`detect.py`**: Utilities for simulating detection events and efficiencies.
- **`lhv_communication_loophole.py`**: LHV model exploiting the communication loophole.
- **`lhv_detection_loophole.py`**: LHV model exploiting the detection loophole (Garg-Mermin).
- **`lhv_memory_loophole.py`**: LHV model exploiting the memory loophole (time-dependence).
- **`noise.py`**: Implements noise models (depolarization, dark counts).
- **`qiskit_backend.py`**: Qiskit-based simulation backend.
- **`__init__.py`**: Module initialization.

## `stats/` (Statistical Utilities)
- **`bootstrap.py`**: Generic bootstrap resampling functions.
- **`fdr.py`**: False Discovery Rate control (Benjamini-Hochberg).
- **`intervals.py`**: Confidence interval calculations (Clopper-Pearson, etc.).
- **`power.py`**: Statistical power analysis tools.
- **`sprt.py`**: Sequential Probability Ratio Test (Martingale tests).
- **`__init__.py`**: Module initialization.

## `viz/` (Visualization)
- **`overlays.py`**: Helper functions for overlaying plots.
- **`plots.py`**: Core plotting functions for histograms, ROC curves, and heatmaps.
- **`__init__.py`**: Module initialization.

## `witness/` (Entanglement Witnesses)
- **`cew.py`**: Core implementation of Conformal Entanglement Witnesses.
- **`cew_calib.py`**: Calibration routines for CEW.
- **`cew_entropic.py`**: Entropic uncertainty-based witnesses.
- **`cew_improved.py`**: Optimized CEW implementation.
- **`cmi.py`**: Contextual Miscalibration Index (CMI) calculation.
- **`mlw.py`**: Machine Learning Witness (Scenario A & B) implementation.
- **`sheaf.py`**: Sheaf-theoretic Contextual Fraction calculations.
- **`__init__.py`**: Module initialization.

## `dashboard/` (Web Interface)
- **`app.py`**: Streamlit application for interactive data visualization.
