import numpy as np
from src.stats.intervals import wilson_interval
from src.stats.fdr import benjamini_hochberg


def test_wilson_basic():
    lo, hi = wilson_interval(50, 100, alpha=0.05)
    assert 0.35 < lo < 0.5
    assert 0.5 < hi < 0.65


def test_bh_monotone():
    reject, cutoff = benjamini_hochberg([0.001, 0.01, 0.2, 0.5], q=0.1)
    assert reject.sum() >= 1
