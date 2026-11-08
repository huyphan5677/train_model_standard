import numpy as np
import pytest

from train_model.scoring import fit_score_bounds, proba_to_score


@pytest.fixture
def proba():
    return np.random.default_rng(0).uniform(0.001, 0.999, 2000)


def test_score_range_and_monotonic(proba):
    s = proba_to_score(proba)
    assert s.dtype.kind == "i"
    assert s.min() >= 300 and s.max() <= 850
    order = np.argsort(proba)
    assert np.all(np.diff(s[order]) >= 0)
    assert s.min() == 300 and s.max() == 850


def test_score_bad_label_inverts(proba):
    good = proba_to_score(proba, target_label="good")
    bad = proba_to_score(proba, target_label="bad")
    assert np.corrcoef(good, bad)[0, 1] < -0.99


def test_score_fixed_bounds_stable(proba):
    bounds = fit_score_bounds(proba)
    full = proba_to_score(proba, bounds=bounds)
    subset = proba_to_score(proba[:100], bounds=bounds)
    np.testing.assert_array_equal(full[:100], subset)


def test_score_invalid_label(proba):
    with pytest.raises(ValueError):
        proba_to_score(proba, target_label="neutral")
