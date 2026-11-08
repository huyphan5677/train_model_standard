import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from train_model.evaluation import auc_gini, logistic_ablation, refit_ablation
from train_model.feature_selection import (
    Condition,
    ShadowFeatureSelector,
    TreeRuleMiner,
    extract_rules,
)
from train_model.feature_selection.rules import simplify
from train_model.modeling import LogisticScorecard


# ----------------------------------------------------------------------------- shadow
def test_shadow_keeps_signal_drops_noise():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f"s{i}": rng.normal(size=n) for i in range(3)} | {f"n{i}": rng.normal(size=n) for i in range(3)})
    X["cat_sig"] = np.where(rng.random(n) < 0.5, "x", "y")
    logit = X["s0"] + 0.8 * X["s1"] + 0.6 * X["s2"] + np.where(X["cat_sig"] == "x", 0.8, 0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    sel = ShadowFeatureSelector(n_shadows=2, cv=2, catboost_params={"iterations": 80}).fit(X, y)
    assert {"s0", "s1", "s2", "cat_sig"} <= set(sel.selected_features_)
    assert len(set(sel.selected_features_) & {"n0", "n1", "n2"}) <= 1
    assert {"mean_importance", "threshold", "share_runs_above", "keep"} <= set(sel.report_.columns)
    assert sel.runs_.shape == (7, 4)
    assert list(sel.transform(X).columns) == sel.selected_features_


# ----------------------------------------------------------------------------- ablation
@pytest.fixture(scope="module")
def lr_data():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"strong": rng.normal(size=n), "mid": rng.normal(size=n), "weak": rng.normal(size=n),
                      "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    X["mid_corr"] = X["mid"] + rng.normal(0, 0.5, n)
    logit = -0.5 + 1.2 * X["strong"] + 0.6 * X["mid"] + 0.15 * X["weak"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X.iloc[:4000], y[:4000], X.iloc[4000:], y[4000:]


@pytest.mark.parametrize("C", [1.0, 0.05, None])
def test_logistic_ablation_matches_real_refit(lr_data, C):
    Xtr, ytr, Xev, yev = lr_data
    kw = {"penalty": None} if C is None else {"C": C}
    lr = LogisticRegression(max_iter=5000, **kw).fit(Xtr, ytr)
    res = logistic_ablation(Xtr, ytr, lr.coef_[0], lr.intercept_[0], Xev, yev, C=C).set_index("feature")
    for f in Xtr.columns:
        cols = [c for c in Xtr.columns if c != f]
        p = LogisticRegression(max_iter=5000, **kw).fit(Xtr[cols], ytr).predict_proba(Xev[cols])[:, 1]
        assert res.loc[f, "gini_without"] == pytest.approx(auc_gini(yev, p)["gini"], abs=2e-3)
    assert res.loc["strong", "significant"] and not res.loc["noise1", "significant"]
    assert res.index[0] == "strong"  # largest loss first


def test_refit_ablation_generic(lr_data):
    Xtr, ytr, Xev, yev = lr_data

    def fit_predict(cols):
        return LogisticRegression(max_iter=5000).fit(Xtr[cols], ytr).predict_proba(Xev[cols])[:, 1]

    res = refit_ablation(fit_predict, list(Xtr.columns), yev)
    assert len(res) == Xtr.shape[1] and res.iloc[0]["feature"] == "strong"
    assert (res["gini_full"] == res["gini_full"].iloc[0]).all()


# ----------------------------------------------------------------------------- rules
def test_simplify_merges_bounds_and_dedupes():
    r1 = simplify([Condition("a", "<=", 2.0), Condition("b", ">", 0.0), Condition("a", "<=", 1.0)])
    r2 = simplify([Condition("b", ">", 0.0), Condition("a", "<=", 1.0)])
    assert r1 == r2 and len(r1) == 2 and str(r1) == "a <= 1 & b > 0"


def test_extract_rules_from_tree():
    X = pd.DataFrame({"a": [0, 0, 1, 1] * 25, "b": [0, 1, 0, 1] * 25}, dtype=float)
    y = ((X["a"] == 1) & (X["b"] == 1)).astype(int)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    rules = extract_rules([tree], ["a", "b"])
    assert len(rules) == tree.get_n_leaves()  # one rule per leaf
    hit = [r for r in rules if r.apply(X).sum() and y[r.apply(X)].all()]
    assert len(hit) == 1 and hit[0].features == ["a", "b"]


@pytest.fixture(scope="module")
def ixn_data():
    rng = np.random.default_rng(0)
    n = 8000
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    both = (X["x1"] > 0.3) & (X["x2"] > 0.3)
    logit = -1.0 + 0.4 * X["x3"] + 2.0 * both
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, pd.Series(y)


def test_miner_finds_the_interaction(ixn_data):
    X, y = ixn_data
    miner = TreeRuleMiner(n_estimators=50, max_depth=2, min_target_support=0.05).fit(X, y)
    assert miner.rules_, miner.n_rules_
    assert any(set(r.features) == {"x1", "x2"} for r in miner.rules_.values())
    counts = list(miner.n_rules_.values())
    assert counts == sorted(counts, reverse=True)  # each step only removes
    out = miner.transform(X)
    assert list(out.columns) == list(miner.rules_) and set(np.unique(out.to_numpy())) <= {0, 1}
    assert {"name", "rule", "length", "support", "target_rate", "n_target", "lasso_coef"} == set(miner.stats_.columns)
    with pytest.raises(ValueError):
        TreeRuleMiner().fit(X.assign(x1=np.nan), y)


def test_miner_with_base_model_ignores_additive_data():
    """Regression: with no real interaction, rules mined on the residual of a logistic model add nothing."""
    rng = np.random.default_rng(1)
    n = 6000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["a"] + 0.7 * X["b"] - 0.5 * X["c"])))).astype(int)
    with_base = TreeRuleMiner(base_estimator=LogisticRegression(), min_target_support=0.05).fit(X, y)
    without = TreeRuleMiner(min_target_support=0.05).fit(X, y)
    assert len(with_base.rules_) <= 2 < len(without.rules_)


def test_scorecard_with_interactions(ixn_data):
    X, y = ixn_data
    tr, te = slice(0, 6000), slice(6000, None)
    params = {"n_estimators": 50, "max_depth": 2, "min_target_support": 0.05}
    plain = LogisticScorecard(max_n_bins=5).fit(X[tr], y[tr])
    rich = LogisticScorecard(max_n_bins=5, interactions=True, interaction_params=params).fit(X[tr], y[tr])
    g_plain = auc_gini(y[te], plain.predict_proba(X[te])[:, 1])["gini"]
    g_rich = auc_gini(y[te], rich.predict_proba(X[te])[:, 1])["gini"]
    assert g_rich > g_plain + 0.01
    table = rich.points_table()
    rule_rows = table[table["feature"].str.startswith("ixn_")]
    assert len(rule_rows) == len(rich.miner_.rules_) and rule_rows["bin"].str.contains(" in {").all()
    p = rich.predict_proba(X[te])[:, 1]
    factor = rich.pdo / np.log(2)
    expected = rich.base_score - factor * np.log(rich.base_odds) + factor * np.log(p / (1 - p))
    np.testing.assert_allclose(rich.points(X[te]), expected, rtol=1e-6)
    abl = rich.ablation(X[tr], y[tr], X[te], y[te])
    assert set(abl["feature"]) == set(rich.design_columns_)
