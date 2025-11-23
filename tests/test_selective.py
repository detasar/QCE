import numpy as np
from src.cp.selective import selective_coverage


def test_selective_coverage():
    y_true = np.array([1, 0, 1, 1, 0])
    in_set = np.array([True, True, False, True, False])
    abstain = np.array([False, True, False, False, True])
    cov = selective_coverage(y_true, in_set, abstain)
    assert 0 <= cov <= 1
