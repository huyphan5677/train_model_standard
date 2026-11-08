import numpy as np
import pandas as pd
import pytest

from tests.conftest import CAT_COLS, ID_COLS
from train_model.data import INFORMATIVE_FEATURES
from train_model.feature_selection import (
    HardFeatureSelector,
    WoETransformer,
    compute_iv,
    fit_optbinning,
)


@pytest.fixture(scope="module")
def train_xy(splits):
    train, _, _ = splits
    X = train.drop(columns=ID_COLS)
    return X, train["GB"]


@pytest.fixture(scope="module")
def fitted(train_xy):
    X, y = train_xy
    return HardFeatureSelector(cat_cols=CAT_COLS).fit(X, y)


def _step(report, feature):
    rows = report.loc[report["feature"] == feature, "step"]
    return rows.iloc[0] if len(rows) else None


def test_hard_selector_removes_traps(fitted):
    rep = fitted.report_
    assert _step(rep, "all_missing") == "missing"
    assert _step(rep, "mostly_missing") == "missing"
    assert _step(rep, "const_col") == "constant"
    assert _step(rep, "dup_of_usage") == "duplicate"
    corr_losers = {f for f in ("recharge_amt_l1m", "corr_of_recharge") if _step(rep, f) == "high_corr"}
    assert len(corr_losers) == 1
    kept_informative = set(INFORMATIVE_FEATURES) & set(fitted.selected_features_)
    assert set(INFORMATIVE_FEATURES) - kept_informative <= corr_losers
    for noise in (f"noise_{i}" for i in range(1, 6)):
        assert noise not in fitted.selected_features_ or fitted.iv_[noise] >= 0.02


def test_transform_returns_raw_selected_columns(fitted, train_xy):
    X, _ = train_xy
    out = fitted.transform(X)
    assert list(out.columns) == fitted.selected_features_
    pd.testing.assert_series_equal(out["usage_data_mb_l1m"], X["usage_data_mb_l1m"])


def test_hard_selector_does_not_mutate_input(train_xy):
    X, y = train_xy
    before = X.copy()
    HardFeatureSelector(cat_cols=CAT_COLS).fit(X, y)
    pd.testing.assert_frame_equal(X, before)


def test_hard_selector_handles_cat_with_nan():
    rng = np.random.default_rng(1)
    n = 800
    y = pd.Series(rng.integers(0, 2, n))
    cat = np.where((y == 1) & (rng.random(n) < 0.7), "good", rng.choice(["a", "b"], n)).astype(object)
    cat[rng.random(n) < 0.1] = None
    X = pd.DataFrame({"cat": cat, "num": rng.normal(size=n) + y})
    sel = HardFeatureSelector(cat_cols=["cat"]).fit(X, y)
    assert "cat" in sel.selected_features_


def test_hard_selector_drops_all_nan_column():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 500))
    X = pd.DataFrame({"empty": np.nan, "good": rng.normal(size=500) + 2 * y})
    sel = HardFeatureSelector(missing_threshold=1.0).fit(X, y)
    assert sel.selected_features_ == ["good"]
    assert _step(sel.report_, "empty") in {"constant", "low_iv", "missing"}


def test_two_selectors_independent(train_xy):
    X, y = train_xy
    a = HardFeatureSelector(cat_cols=CAT_COLS).fit(X, y)
    iv_before = a.iv_.copy()
    HardFeatureSelector().fit(X[["noise_1", "noise_2"]], y)
    pd.testing.assert_series_equal(a.iv_, iv_before)


def test_compute_iv_informative_higher_than_noise(train_xy):
    X, y = train_xy
    feats = list(INFORMATIVE_FEATURES) + [f"noise_{i}" for i in range(1, 6)]
    iv = compute_iv(X.assign(GB=y), feats, "GB").set_index("feature")["iv"]
    assert iv[list(INFORMATIVE_FEATURES)].mean() > iv[[f"noise_{i}" for i in range(1, 6)]].mean()


def test_woe_transformer(train_xy):
    X, y = train_xy
    cols = ["recharge_amt_l1m", "device_manu"]
    woe = WoETransformer(features=cols, cat_cols=["device_manu"]).fit(X, y)
    out = woe.transform(X)
    assert out[cols].dtypes.map(lambda d: np.issubdtype(d, np.number)).all()
    assert out["recharge_amt_l1m"].nunique() <= 11
    assert "device_manu" in fit_optbinning(X.assign(GB=y), cols, "GB", cat_cols=["device_manu"])
