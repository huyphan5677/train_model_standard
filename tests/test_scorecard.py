import numpy as np
import pandas as pd
import pytest

from train_model.data import INFORMATIVE_FEATURES
from train_model.evaluation import auc_gini
from train_model.modeling import LogisticScorecard

FEATURES = [*INFORMATIVE_FEATURES, "device_manu", "sub_type"]
CATS = ["device_manu", "sub_type"]


@pytest.fixture(scope="module")
def fitted(splits):
    train, _, test = splits
    sc = LogisticScorecard(features=FEATURES, cat_cols=CATS, max_n_bins=5).fit(train, train["GB"])
    return sc, train, test


def test_scorecard_discriminates(fitted):
    sc, _, test = fitted
    proba = sc.predict_proba(test)[:, 1]
    assert auc_gini(test["GB"], proba)["gini"] > 0.3
    assert set(sc.predict(test)) <= {0, 1}
    assert list(sc.classes_) == [0, 1]


def test_points_follow_pdo_scaling(fitted):
    sc, _, test = fitted
    p = sc.predict_proba(test)[:, 1]
    pts = sc.points(test)
    # label 1 = good -> points = offset + factor * ln(p / (1 - p))
    expected = sc.base_score - sc.pdo / np.log(2) * np.log(sc.base_odds) + sc.pdo / np.log(2) * np.log(p / (1 - p))
    np.testing.assert_allclose(pts, expected, rtol=1e-6)
    assert pd.Series(pts).corr(pd.Series(p), method="spearman") > 0.999


def test_points_table_adds_up(fitted):
    sc, _, test = fitted
    table = sc.points_table()
    assert table.iloc[0]["feature"] == "(base)"
    assert set(table["feature"]) - {"(base)"} == set(sc.features_)
    rows = test.head(50)
    total = np.full(len(rows), table.iloc[0]["points"])
    for f in sc.features_:
        optb = sc.woe_.binnings_[f]
        is_cat = optb.dtype == "categorical"
        x = rows[f].astype(object).where(rows[f].notna(), None).map(lambda v: v if v is None else str(v)) if is_cat \
            else pd.to_numeric(rows[f], errors="coerce")
        woe = optb.transform(x.to_numpy(), metric="woe", metric_missing="empirical")
        total += sc.pdo / np.log(2) * sc.coef_[f] * woe
    np.testing.assert_allclose(sc.points(rows), total, rtol=1e-6)


def test_bad_label_flips_points(splits):
    train, _, test = splits
    good = LogisticScorecard(features=list(INFORMATIVE_FEATURES), max_n_bins=5).fit(train, train["GB"])
    bad = LogisticScorecard(features=list(INFORMATIVE_FEATURES), max_n_bins=5, target_label="bad").fit(
        train, train["GB"]
    )
    rho = pd.Series(good.points(test)).corr(pd.Series(bad.points(test)), method="spearman")
    assert rho < -0.999
    with pytest.raises(ValueError):
        LogisticScorecard(target_label="x").fit(train[list(INFORMATIVE_FEATURES)], train["GB"])
