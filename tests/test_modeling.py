import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from tests.conftest import CAT_COLS
from train_model.data import INFORMATIVE_FEATURES
from train_model.modeling import (
    CatBoostEnsembleClassifier,
    CatBoostShrinker,
    CatBoostTransferLearner,
    ModelBundle,
    evaluate_splits,
    train_catboost,
)

FEATURES = list(INFORMATIVE_FEATURES) + CAT_COLS
PARAMS = {"iterations": 150, "depth": 4, "learning_rate": 0.1, "eval_metric": "AUC"}


@pytest.fixture(scope="module")
def trained(splits):
    train, val, _ = splits
    return train_catboost(train, val, FEATURES, "GB", cat_features=CAT_COLS, params=PARAMS)


def test_train_catboost_learns(trained, splits):
    _, _, test = splits
    gini = 2 * roc_auc_score(test["GB"], trained.predict_proba(test[FEATURES])[:, 1]) - 1
    assert gini > 0.3


def test_train_with_cat_nan(synthetic_df):
    raw = synthetic_df.head(800)  # raw categorical columns still contain None / ""
    model = train_catboost(raw.iloc[:600], raw.iloc[600:], FEATURES, "GB", cat_features=CAT_COLS,
                           params={"iterations": 20})
    assert model.tree_count_ > 0


def test_evaluate_splits_no_mutation(trained, splits):
    train, val, test = splits
    before = test.copy()
    res = evaluate_splits(trained, {"train": train, "val": val, "test": test}, FEATURES, "GB")
    assert list(res.columns) == ["split", "n", "auc", "gini"]
    assert res["split"].tolist() == ["train", "val", "test"]
    pd.testing.assert_frame_equal(test, before)


def test_bundle_roundtrip_predictions_equal(trained, splits, tmp_path):
    _, _, test = splits
    bundle = ModelBundle(
        model=trained, features=FEATURES, cat_features=CAT_COLS, target="GB", model_code="demo",
        cat_fill="unknown", num_fill=None, train_period=(20240801, 20250201), score_bounds=(-3.0, 4.0),
        metrics={"gini_test": 0.5},
    )
    path = bundle.save(tmp_path / "demo")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    assert meta["features"] == FEATURES and meta["model_code"] == "demo"

    loaded = ModelBundle.load(path)
    assert loaded.features == FEATURES and loaded.cat_features == CAT_COLS
    assert loaded.cat_fill == "unknown" and loaded.num_fill is None
    assert loaded.score_bounds == (-3.0, 4.0) and loaded.train_period == (20240801, 20250201)
    np.testing.assert_allclose(loaded.predict_proba(test), bundle.predict_proba(test))


def test_bundle_predict_normalizes_raw_categories(trained, synthetic_df):
    bundle = ModelBundle(model=trained, features=FEATURES, cat_features=CAT_COLS, target="GB",
                         model_code="demo", cat_fill="unknown", num_fill=None)
    raw = synthetic_df.head(50).copy()
    raw.loc[0, "device_manu"] = "  APPLE "
    proba = bundle.predict_proba(raw)
    assert proba.shape == (50,) and np.all((proba >= 0) & (proba <= 1))


def test_ensemble_and_shrinker(splits, tmp_path):
    train, val, _ = splits
    X, y = train[list(INFORMATIVE_FEATURES)], train["GB"]
    Xv, yv = val[list(INFORMATIVE_FEATURES)], val["GB"]
    m1 = CatBoostClassifier(iterations=40, verbose=False, random_seed=1).fit(X, y)
    m2 = CatBoostClassifier(iterations=40, verbose=False, random_seed=2).fit(X, y)

    ens = CatBoostEnsembleClassifier([m1, m2], weights=[2, 1])
    proba = ens.predict_proba(Xv)
    assert proba.shape == (len(Xv), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    merged = ens.merge_to_single_model()
    assert merged.tree_count_ == 80

    shrinker = CatBoostShrinker(m1)
    shrinker.shrink_by_validation(Xv, yv, roc_auc_score, step=10)
    assert shrinker.best_ntree_end <= m1.tree_count_
    assert shrinker.get_shrunk_model().tree_count_ == shrinker.best_ntree_end
    assert m1.tree_count_ == 40  # original untouched

    tl = CatBoostTransferLearner(m1)
    tl.fine_tune(Xv, yv, iterations=10)
    assert tl.model.tree_count_ == 50
    tl.save(tmp_path / "tl.cbm", tmp_path / "tl.json")
    assert CatBoostTransferLearner.load(tmp_path / "tl.json").model.tree_count_ == 50
