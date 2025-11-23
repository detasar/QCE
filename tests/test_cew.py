from src.witness.cew import threshold_L, cardinality_product


def test_cew_threshold_safe():
    L = threshold_L(0.2, 0.7071067811865476, {'L_formula': '(1-alpha)**2 / c'})
    assert L > 0


def test_cew_decision_logic():
    prod = cardinality_product(1.0, 1.2)
    L = threshold_L(0.2, 0.7071067811865476, {'L_formula': '(1-alpha)**2 / c'})
    assert isinstance(prod, float)
    assert isinstance(L, float)
