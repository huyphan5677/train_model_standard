import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from tests.conftest import CAT_COLS
from train_model.data import INFORMATIVE_FEATURES
from train_model.feature_selection import (
    ALL_CRITERIA,
    ImportanceSelector,
    SoftFeatureSelector,
    best_k,
    forward_selection_gini,
    select_by_importance,
)

NOISE = [f"noise_{i}" for i in range(1, 6)]
FEATURES = list(INFORMATIVE_FEATURES) + NOISE + CAT_COLS


@pytest.fixture(scope="module")
def data(splits):
    train, val, _ = splits
    return train[FEATURES], train["GB"], val[FEATURES], val["GB"]


@pytest.fixture(scope="module")
def soft(data):
    X, y, Xv, yv = data
    return SoftFeatureSelector(top_k=7, min_votes=3, cat_cols=CAT_COLS, n_estimators=50).fit(X, y, Xv, yv)


def test_soft_selector_votes_shape(soft):
    assert set(soft.votes_.columns) == set(ALL_CRITERIA) | {"n_votes"}
    assert (soft.votes_["n_votes"] == soft.votes_[list(ALL_CRITERIA)].sum(axis=1)).all()
    assert set(soft.votes_.index) == set(FEATURES)
    assert soft.votes_["n_votes"].is_monotonic_decreasing


def test_soft_selector_prefers_informative(soft):
    votes = soft.votes_["n_votes"]
    assert votes[list(INFORMATIVE_FEATURES)].mean() > votes[NOISE].mean() + 2


def test_min_votes_filters(soft, data):
    assert all(soft.votes_.loc[f, "n_votes"] >= 3 for f in soft.selected_)
    X, y, Xv, yv = data
    none = SoftFeatureSelector(
        top_k=7, min_votes=len(ALL_CRITERIA) + 1, n_estimators=20, cat_cols=CAT_COLS
    ).fit(X, y, Xv, yv)
    assert none.selected_ == []
    assert list(soft.transform(X).columns) == soft.selected_


def test_soft_selector_does_not_mutate_input(data):
    X, y, Xv, yv = data
    before, before_v = X.copy(), Xv.copy()
    SoftFeatureSelector(top_k=5, n_estimators=20, cat_cols=CAT_COLS, criteria=["tree_vs_random", "l1"]).fit(
        X, y, Xv, yv
    )
    pd.testing.assert_frame_equal(X, before)
    pd.testing.assert_frame_equal(Xv, before_v)


def test_subset_criteria_and_unknown_raises(data):
    X, y, Xv, yv = data
    sel = SoftFeatureSelector(criteria=["correlation", "variance"], cat_cols=CAT_COLS).fit(X, y, Xv, yv)
    assert set(sel.votes_.columns) == {"correlation", "variance", "n_votes"}
    with pytest.raises(ValueError, match="bogus"):
        SoftFeatureSelector(criteria=["bogus"])


def test_correlation_uses_absolute_value():
    rng = np.random.default_rng(3)
    n = 600
    y = pd.Series(rng.integers(0, 2, n))
    X = pd.DataFrame({"neg": -3.0 * y + rng.normal(size=n), "weak": 0.1 * y + rng.normal(size=n)})
    sel = SoftFeatureSelector(top_k=1, min_votes=1, criteria=["correlation"]).fit(X, y, X, y)
    assert sel.selected_ == ["neg"]


def test_forward_selection_and_best_k(data):
    X, y, Xv, yv = data
    res = forward_selection_gini(
        X, y, Xv, yv, ordered_features=list(INFORMATIVE_FEATURES[:5]), catboost_params={"iterations": 50}
    )
    assert list(res.columns) == ["n_features", "last_feature", "gini_val"]
    assert len(res) == 5 and res["n_features"].tolist() == [1, 2, 3, 4, 5]
    manual = pd.DataFrame({"n_features": [1, 2, 3, 4, 5], "gini_val": [0.30, 0.40, 0.452, 0.455, 0.454]})
    assert best_k(manual, tolerance=0.005) == 3


@pytest.mark.parametrize(
    "estimator",
    [CatBoostClassifier(iterations=50, verbose=False), RandomForestClassifier(n_estimators=50, random_state=0)],
    ids=["catboost", "random_forest"],
)
def test_select_by_importance_accumulate_normalized(data, estimator):
    X, y, _, _ = data
    Xn = X.drop(columns=CAT_COLS)
    selected, imp = select_by_importance(Xn, y, estimator, accumulate=0.9)
    assert imp["importance"].sum() == pytest.approx(1.0)
    assert 0 < len(selected) < Xn.shape[1]
    assert imp.set_index("feature").loc[selected[:-1], "cumulative"].max() <= 0.9 + 1e-9
    with pytest.raises(ValueError):
        select_by_importance(Xn, y, estimator)


def test_importance_selector_top_k(data):
    X, y, _, _ = data
    Xn = X.drop(columns=CAT_COLS)
    sel = ImportanceSelector(top_k=4).fit(Xn, y)
    assert sel.transform(Xn).shape[1] == 4
