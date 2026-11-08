import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from matplotlib.figure import Figure

from train_model.data import INFORMATIVE_FEATURES
from train_model.evaluation import (
    auc_gini,
    auc_gini_by_group,
    decile_table,
    plot_forward_gini,
    plot_roc,
    plot_roc_by_group,
    plot_shap_summary,
    pr_auc_by_bin,
    psi,
    psi_by_bucket,
    psi_by_feature,
    shap_group_importance,
    shap_importance,
)


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    score = np.clip(0.5 * y + rng.normal(0.25, 0.2, n), 0, 1)
    return pd.DataFrame({"y": y, "p": score, "g": rng.choice(["A", "B"], n)})


@pytest.fixture(scope="module")
def model_and_x(splits):
    train, _, _ = splits
    X = train[list(INFORMATIVE_FEATURES)]
    model = CatBoostClassifier(iterations=50, verbose=False, random_seed=0).fit(X, train["GB"])
    return model, X


def test_auc_gini_perfect():
    res = auc_gini([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert res == {"auc": 1.0, "gini": 1.0}


def test_auc_gini_by_group_single_class_nan(scored):
    df = scored.copy()
    df.loc[df["g"] == "B", "y"] = 1
    res = auc_gini_by_group(df, "y", "p", "g").set_index("group")
    assert np.isnan(res.loc["B", "auc"]) and not np.isnan(res.loc["A", "auc"])
    assert res.loc["A", "n"] == (df["g"] == "A").sum()


def test_decile_table_monotonic(scored):
    t = decile_table(scored["y"], scored["p"])
    assert len(t) == 10
    assert t["count"].sum() == len(scored)
    assert t["acc_positive"].iloc[-1] == scored["y"].sum()
    assert t["top"].tolist() == list(range(1, 11))
    assert t["positive_rate"].iloc[0] > t["positive_rate"].iloc[-1]
    assert t["cover_positive"].iloc[-1] == pytest.approx(1.0)


def test_decile_table_with_ties():
    y = np.array([0, 1] * 50)
    p = np.repeat([0.1, 0.5, 0.9], [40, 30, 30])
    t = decile_table(y, p)
    assert len(t) <= 3
    assert t["count"].sum() == 100


def test_decile_table_with_score(scored):
    t = decile_table(scored["y"], scored["p"], score=scored["p"] * 1000)
    assert "score_range" in t.columns


def test_pr_auc_by_bin(scored):
    t = pr_auc_by_bin(scored["y"], scored["p"], n_bins=5)
    assert list(t.columns) == ["bin", "pr_auc"] and len(t) == 5


def test_psi_identical_zero_shifted_positive():
    rng = np.random.default_rng(1)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-9
    assert psi(a, a + 1.0) > 0.1
    assert psi(a, a + 1.0, method="uniform") > 0.1


def test_psi_constant_and_empty():
    assert np.isfinite(psi(np.ones(100), np.ones(50)))
    assert np.isnan(psi([], [1, 2, 3]))
    assert np.isnan(psi([np.nan, np.nan], [1.0]))


def test_psi_by_feature_handles_categorical():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"num": rng.normal(size=1000), "cat": rng.choice(["a", "b"], 1000, p=[0.8, 0.2])})
    cur = pd.DataFrame({"num": rng.normal(size=1000), "cat": rng.choice(["a", "b", "c"], 1000, p=[0.3, 0.3, 0.4])})
    res = psi_by_feature(ref, cur, ["num", "cat"]).set_index("feature")["psi"]
    assert res["num"] < 0.05
    assert res["cat"] > 0.25


def test_psi_by_bucket():
    ref = np.array([350, 450, 550, 650, 750] * 20)
    cur = np.array([350, 350, 450, 550, 750] * 20)
    t = psi_by_bucket(ref, cur, [300, 400, 500, 600, 700, 850])
    assert list(t.columns) == ["bucket", "ref_pct", "cur_pct", "psi"]
    assert t["ref_pct"].sum() == pytest.approx(1.0)


def test_shap_importance_columns(model_and_x):
    model, X = model_and_x
    imp = shap_importance(model, X, max_samples=300)
    assert list(imp.columns) == ["feature", "mean_abs_shap"]
    assert len(imp) == X.shape[1] and (imp["mean_abs_shap"] >= 0).all()
    grp = shap_group_importance(imp).set_index("group")
    assert {"usage", "recharge"} <= set(grp.index)
    assert grp.loc["usage", "count"] == 2


def test_plots_return_figure(scored, model_and_x):
    model, X = model_and_x
    assert isinstance(plot_roc({"test": (scored["y"], scored["p"])}), Figure)
    assert isinstance(plot_roc_by_group(scored, "y", "p", "g"), Figure)
    assert isinstance(plot_shap_summary(model, X.head(200), max_display=5), Figure)
    fwd = pd.DataFrame({"n_features": [1, 2, 3], "last_feature": list("abc"), "gini_val": [0.2, 0.3, 0.31]})
    assert isinstance(plot_forward_gini(fwd), Figure)
