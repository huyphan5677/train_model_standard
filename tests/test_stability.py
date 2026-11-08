import numpy as np
import pandas as pd
import pytest

from train_model.feature_selection import TemporalStabilitySelector, temporal_consistency
from train_model.feature_selection.stability import (
    INSUFFICIENT,
    MISSING_BIN,
    NO_SIGNAL,
    STABLE,
    UNSTABLE,
    apply_bins,
    fit_bin_edges,
)


def _panel(n=12000, n_periods=8, seed=0):
    """Toy panel: ``stable`` keeps its effect, ``flip`` reverses it every other period, ``noise`` has none."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, n_periods, n)
    stable = rng.normal(size=n)
    flip = rng.normal(size=n)
    sign = np.where(t % 2 == 0, 1.0, -1.0)
    logit = -0.5 + 1.0 * stable + 1.2 * sign * flip
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cat = np.where(stable > 0.5, "high", np.where(stable < -0.5, "low", "mid")).astype(object)
    cat[rng.random(n) < 0.05] = None
    X = pd.DataFrame({"t": t, "stable": stable, "flip": flip, "noise": rng.normal(size=n), "cat": cat})
    return X, y


def test_bins_numeric_and_categorical():
    x = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    labels = apply_bins(x, fit_bin_edges(x, n_bins=3))
    assert labels.iloc[3] == MISSING_BIN
    assert labels.drop(3).nunique() == 3
    cat = apply_bins(pd.Series(["a", None, "b"]), None)
    assert cat.tolist() == ["a", MISSING_BIN, "b"]


def test_constant_feature_edges_single_bin():
    edges = fit_bin_edges(pd.Series([1.0] * 10))
    assert apply_bins(pd.Series([1.0, 1.0]), edges).nunique() == 1


def test_stable_vs_flipping_vs_noise():
    X, y = _panel()
    bins = lambda c: apply_bins(X[c], fit_bin_edges(X[c], 5))  # noqa: E731
    stable = temporal_consistency(bins("stable"), y, X["t"])
    flip = temporal_consistency(bins("flip"), y, X["t"])
    noise = temporal_consistency(bins("noise"), y, X["t"])
    assert stable.status == STABLE and stable.score > 0.9
    assert flip.status == UNSTABLE and flip.score < 0.3
    assert noise.status == NO_SIGNAL
    assert {"bin", "t", "count", "rate", "tr", "weight"} <= set(stable.table.columns)
    assert stable.n_periods == 8


def test_weak_but_stable_feature_is_stable():
    """Regression: a weak, time-constant effect must not be flagged unstable because of noisy small cells."""
    rng = np.random.default_rng(3)
    n, n_periods = 5000, 9
    t = rng.integers(0, n_periods, n)
    weak = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.6 + 0.3 * weak)))).astype(int)
    res = temporal_consistency(apply_bins(pd.Series(weak), fit_bin_edges(pd.Series(weak), 10)), y, t)
    assert res.status == STABLE


def test_single_period_is_insufficient():
    X, y = _panel(n=3000)
    res = temporal_consistency(apply_bins(X["stable"], fit_bin_edges(X["stable"], 5)), y, np.zeros(len(y)))
    assert res.status == INSUFFICIENT and np.isnan(res.score)


def test_selector_drops_flipping_feature_and_keeps_stable():
    X, y = _panel()
    sel = TemporalStabilitySelector(time_col="t", n_bins=5, cat_cols=["cat"]).fit(X, y)
    rep = sel.report_.set_index("feature")
    assert "stable" in sel.selected_features_ and "cat" in sel.selected_features_
    assert "flip" not in sel.selected_features_
    assert rep.loc["flip", "status"] == UNSTABLE and not rep.loc["flip", "keep"]
    assert list(sel.transform(X).columns) == sel.selected_features_
    assert set(sel.tables_) == {"stable", "flip", "noise", "cat"}
    assert "t" not in sel.tables_


def test_selector_drop_no_signal_option():
    X, y = _panel()
    keep = TemporalStabilitySelector(time_col="t", n_bins=5).fit(X, y)
    drop = TemporalStabilitySelector(time_col="t", n_bins=5, drop_no_signal=True).fit(X, y)
    assert set(drop.selected_features_) <= set(keep.selected_features_)
    no_signal = keep.report_.loc[keep.report_["status"] == NO_SIGNAL, "feature"]
    assert not set(no_signal) & set(drop.selected_features_)


def test_selector_with_validation_sample():
    X, y = _panel(seed=1)
    X_val, y_val = _panel(n=6000, seed=2)
    sel = TemporalStabilitySelector(time_col="t", n_bins=5).fit(X, y, X_val, y_val)
    assert {"score_val", "status_val"} <= set(sel.report_.columns)
    assert "stable" in sel.selected_features_


def test_selector_validates_inputs():
    X, y = _panel(n=2000)
    with pytest.raises(ValueError, match="time_col"):
        TemporalStabilitySelector(time_col="missing").fit(X, y)
    with pytest.raises(ValueError, match="together"):
        TemporalStabilitySelector(time_col="t").fit(X, y, X_val=X)
