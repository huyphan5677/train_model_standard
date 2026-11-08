import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from train_model.evaluation import (
    calibration_summary,
    calibration_table,
    compare_models,
    cross_audit,
    feature_exposure,
    ks_statistic,
    plot_calibration,
    plot_temporal_consistency,
)
from train_model.evaluation.comparison import _scale


@pytest.fixture(scope="module")
def two_signal():
    """y depends on x1 and x2; model ``partial`` sees x1 only, ``full`` sees both."""
    rng = np.random.default_rng(0)
    n = 20000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    logit_full = -1 + 1.0 * x1 + 1.0 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit_full))).astype(int)
    full = 1 / (1 + np.exp(-logit_full))
    partial = 1 / (1 + np.exp(-(-0.8 + 0.8 * x1)))
    noisy = 1 / (1 + np.exp(-(logit_full + rng.normal(0, 2.5, n))))
    return y, {"full": full, "partial": partial, "noisy": noisy}


def test_ks_statistic():
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert 0 <= ks_statistic([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) <= 1e-9


def test_calibration_calibrated_vs_biased():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.6, 20000)
    y = (rng.random(len(p)) < p).astype(int)
    good = calibration_summary(y, p)
    bad = calibration_summary(y, np.clip(p + 0.2, 0, 1))
    assert good["calibration_score"] >= 0.8 and good["hl_p_value"] > 0.001
    assert bad["calibration_score"] == 0.0 and bad["hl_p_value"] < 1e-6
    t = calibration_table(y, np.clip(p + 0.2, 0, 1))
    assert (t["direction"] == "over").all() and t["count"].sum() == len(y)
    assert len(calibration_table(y, p, strategy="uniform")) <= 10
    with pytest.raises(ValueError):
        calibration_table(y, p, strategy="nope")


def test_cross_audit_finds_blind_spot_of_partial_model(two_signal):
    y, scores = two_signal
    res = cross_audit(y, {"full": scores["full"], "partial": scores["partial"]})
    s = res.summary
    assert s.loc["partial", "audited"] > s.loc["full", "audited"]
    assert s.loc["full", "auditor"] > s.loc["partial", "auditor"]
    assert (res.pairs["total"] >= 0).all()
    assert set(res.pairs.columns) == {"audited", "auditor", "up", "down", "total"}


def test_cross_audit_identical_models_no_divergence(two_signal):
    y, scores = two_signal
    res = cross_audit(y, {"a": scores["full"], "b": scores["full"]})
    assert res.pairs["total"].max() == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cross_audit(y, {"a": scores["full"]})


def test_feature_exposure():
    eff, top = feature_exposure({"a": 1, "b": 1, "c": 1, "d": 1})
    assert eff == pytest.approx(4.0) and top == pytest.approx(0.25)
    eff, top = feature_exposure(pd.Series([10.0, 0.0, 0.0]))
    assert eff == pytest.approx(1.0) and top == pytest.approx(1.0)
    assert np.isnan(feature_exposure([0, 0])[0])


def test_scale_lower_better_handles_zero():
    s = _scale(pd.Series({"a": 0.0, "b": 0.02, "c": 0.04}), higher_better=False)
    assert s["a"] == 1.0 and s["b"] == pytest.approx(0.5) and s["c"] == pytest.approx(0.0)
    s = _scale(pd.Series({"a": 0.01, "b": 0.02}), higher_better=False)
    assert s["a"] == 1.0 and s["b"] == pytest.approx(0.5)
    assert (_scale(pd.Series({"a": 0.0, "b": 0.0}), higher_better=False) == 1.0).all()


def test_compare_models_picks_full_model(two_signal):
    y, scores = two_signal
    imp = {"full": {"x1": 1, "x2": 1}, "partial": {"x1": 1}, "noisy": {"x1": 1, "x2": 1}}
    res = compare_models(y, scores, importances=imp)
    assert res.champion == "full"
    assert res.scaled["composite"].iloc[0] == res.scaled.loc["full", "composite"]
    assert res.scaled.drop(columns="composite").max().max() == pytest.approx(1.0)
    assert {"gini", "ks", "calibration", "audited", "auditor", "exposure", "max_share"} <= set(res.metrics.columns)
    assert res.audit is not None


def test_compare_models_weights_and_missing_importance(two_signal):
    y, scores = two_signal
    res = compare_models(y, scores, importances={"full": {"x1": 1}}, weights={"gini": 5})
    assert "exposure" not in res.scaled.columns  # missing for some models -> left out
    single = compare_models(y, {"only": scores["full"]})
    assert single.champion == "only" and single.audit is None


def test_new_plots_return_figure(two_signal):
    y, scores = two_signal
    assert isinstance(plot_calibration({"full": (y, scores["full"]), "noisy": (y, scores["noisy"])}), Figure)
    table = pd.DataFrame({"bin": ["b0", "b0", "b1", "b1"], "t": [1, 2, 1, 2], "tr": [0.3, 0.2, -0.2, -0.1]})
    assert isinstance(plot_temporal_consistency(table, "x", 0.9), Figure)
    assert isinstance(plot_temporal_consistency(table.iloc[0:0], "x", np.nan), Figure)
