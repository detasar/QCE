# Unit Tests

This directory contains unit tests for the core functionality of the package.

## Files
- **`test_cew.py`**: Tests for Conformal Entanglement Witnesses (CEW) logic, ensuring bounds and coverage.
- **`test_cmi.py`**: Tests for Contextual Miscalibration Index (CMI) calculations, verifying null properties.
- **`test_selective.py`**: Tests for selective conformal prediction, checking validity under rejection.
- **`test_stats.py`**: Tests for statistical utility functions (bootstrap, intervals, FDR).

## Running Tests
To run all tests, use `pytest` from the root directory:
```bash
pytest tests/
```
