import numpy as np
from src.witness.cmi import tv_distance_to_uniform, ks_ad_tests


def test_tv_uniform_zeroish():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    tv = tv_distance_to_uniform(p)
    assert tv < 0.1


def test_ks_pvalue_uniform():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    res = ks_ad_tests(p)
    assert 0 <= res['ks_p'] <= 1
