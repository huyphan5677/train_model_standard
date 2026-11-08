import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from train_model.evaluation import auc_ci, delong_test, pairwise_delong


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x - 0.5)))).astype(int)
    strong = x + rng.normal(0, 0.3, n)
    weak = x + rng.normal(0, 2.0, n)
    return y, strong, weak


def test_auc_matches_sklearn_with_ties(data):
    y, strong, _ = data
    rounded = np.round(strong, 1)  # many ties
    res = delong_test(y, rounded, strong)
    assert res["auc_1"] == pytest.approx(roc_auc_score(y, rounded), abs=1e-12)
    assert res["auc_2"] == pytest.approx(roc_auc_score(y, strong), abs=1e-12)


def test_identical_models_not_significant(data):
    y, strong, _ = data
    res = delong_test(y, strong, strong)
    assert res["p_value"] == 1.0 and res["gini_diff"] == 0.0 and not res["significant"]


def test_strong_beats_weak(data):
    y, strong, weak = data
    res = delong_test(y, strong, weak)
    assert res["significant"] and res["gini_diff"] > 0
    assert res["gini_diff_low"] < res["gini_diff"] < res["gini_diff_high"]
    flipped = delong_test(y, weak, strong)
    assert flipped["z"] == pytest.approx(-res["z"])


def test_tiny_perturbation_not_significant(data):
    y, strong, _ = data
    rng = np.random.default_rng(1)
    assert not delong_test(y, strong, strong + rng.normal(0, 1e-3, len(y)))["significant"]


def test_auc_ci_close_to_bootstrap(data):
    y, strong, _ = data
    ci = auc_ci(y, strong)
    assert ci["auc_low"] < ci["auc"] < ci["auc_high"]
    assert ci["gini"] == pytest.approx(2 * ci["auc"] - 1)
    rng = np.random.default_rng(2)
    boot = []
    for _ in range(200):
        i = rng.integers(0, len(y), len(y))
        boot.append(roc_auc_score(y[i], strong[i]))
    assert ci["auc_se"] == pytest.approx(np.std(boot), rel=0.25)


def test_pairwise_and_validation(data):
    y, strong, weak = data
    table = pairwise_delong(y, {"a": strong, "b": weak, "c": strong * 2})
    assert len(table) == 3 and {"model_1", "model_2", "p_value"} <= set(table.columns)
    with pytest.raises(ValueError):
        delong_test(np.ones(10), np.arange(10), np.arange(10))
    with pytest.raises(ValueError):
        delong_test(y, strong[:-1], weak[:-1])
