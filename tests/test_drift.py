import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier

from train_model.evaluation import event_psi, psi_report, shap_drift


def _sample(n, seed, bad_high: bool):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    p = 0.05 + 0.5 * (x if bad_high else 1 - x)
    return pd.DataFrame({"x": x, "cat": rng.choice(["a", "b", "c"], n), "y": (rng.random(n) < p).astype(int)})


def test_event_psi_catches_risk_moving_without_volume_shift():
    ref, cur = _sample(20000, 0, True), _sample(20000, 1, False)
    rep = psi_report(ref, cur, ["x"], "y").set_index("feature")
    assert rep.loc["x", "psi"] < 0.01  # same distribution of x
    assert rep.loc["x", "event_psi"] > 0.25  # but the bads moved to the other end
    assert rep.loc["x", "psi_status"] == "stable" and rep.loc["x", "event_psi_status"] == "critical"


def test_event_psi_same_population_small():
    a, b = _sample(20000, 0, True), _sample(20000, 1, True)
    assert event_psi(a["x"], a["y"], b["x"], b["y"]) < 0.02
    assert event_psi(a["cat"], a["y"], b["cat"], b["y"]) < 0.02


def test_event_label_selects_the_class_followed():
    ref, cur = _sample(20000, 0, True), _sample(20000, 1, False)
    ones = event_psi(ref["x"], ref["y"], cur["x"], cur["y"])
    zeros = event_psi(ref["x"], ref["y"], cur["x"], cur["y"], event_label=0)
    assert ones > 0.25 and 0 < zeros < ones  # zeros are the majority class: their distribution moves less
    rep = psi_report(ref, cur, ["x"], "y", event_label=0).set_index("feature")
    assert rep.loc["x", "event_psi"] == pytest.approx(zeros)


def test_event_psi_handles_empty_cases():
    assert np.isnan(event_psi([1, 2, 3], [0, 0, 0], [1, 2, 3], [0, 0, 0]))


def test_psi_report_weighted_model_row():
    ref, cur = _sample(5000, 0, True), _sample(5000, 1, False)
    rep = psi_report(ref, cur, ["x", "cat"], "y", importance={"x": 3.0, "cat": 1.0})
    assert rep.iloc[-1]["feature"] == "(model)"
    body = rep.iloc[:-1].set_index("feature")
    assert body.loc["x", "weight"] == pytest.approx(0.75)
    assert rep.iloc[-1]["weighted_event_psi"] == pytest.approx(body["weighted_event_psi"].sum())


@pytest.fixture(scope="module")
def shap_model():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    y = (rng.random(n) < 1 / (1 + np.exp(-(X["a"] + 0.5 * X["b"])))).astype(int)
    return CatBoostClassifier(iterations=100, depth=3, verbose=False, random_seed=0).fit(X, y), X


def test_shap_drift_flags_changed_feature(shap_model):
    model, X = shap_model
    same = shap_drift(model, X.iloc[:1500], X.iloc[1500:])
    assert not same["significant"].any()
    shifted = X.iloc[1500:].assign(a=lambda d: d["a"] * 3)
    res = shap_drift(model, X.iloc[:1500], shifted).set_index("feature")
    assert res.loc["a", "significant"] and res.loc["a", "delta"] > 0
    assert list(res.columns) == ["ref_mean_abs_shap", "cur_mean_abs_shap", "delta", "rel_delta", "t", "p_value",
                                 "significant"]
