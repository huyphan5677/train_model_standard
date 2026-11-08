"""Spark integration tests — skipped when pyspark (and a JVM) are not available."""

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from tests.conftest import CAT_COLS  # noqa: E402
from train_model.data import INFORMATIVE_FEATURES  # noqa: E402
from train_model.feature_selection import fit_optbinning  # noqa: E402
from train_model.modeling import ModelBundle, train_catboost  # noqa: E402
from train_model.scoring import fit_score_bounds, proba_to_score  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from train_model.spark.session import create_spark_session

    s = create_spark_session("train_model_tests", master="local[2]")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def bundle(splits):
    train, val, _ = splits
    feats = list(INFORMATIVE_FEATURES) + CAT_COLS
    model = train_catboost(train, val, feats, "GB", cat_features=CAT_COLS, params={"iterations": 50})
    return ModelBundle(model=model, features=feats, cat_features=CAT_COLS, target="GB", model_code="t")


def test_proba_to_score_spark_matches_pandas(spark):
    from train_model.spark.scoring import proba_to_score_spark

    proba = np.random.default_rng(0).uniform(0.01, 0.99, 200)
    bounds = fit_score_bounds(proba)
    sdf = spark.createDataFrame(pd.DataFrame({"id": range(200), "predict_proba": proba}))
    got = proba_to_score_spark(sdf, bounds=bounds).orderBy("id").toPandas()["score"].to_numpy()
    np.testing.assert_array_equal(got, proba_to_score(proba, bounds=bounds))


def test_predict_spark_matches_bundle(spark, bundle, splits):
    from train_model.spark.predict import predict_spark

    _, _, test = splits
    pdf = test.head(300)
    sdf = spark.createDataFrame(pdf)
    out = predict_spark(spark, bundle, sdf, id_col="msisdn").toPandas().set_index("msisdn")
    expected = pd.Series(bundle.predict_proba(pdf), index=pdf["msisdn"])
    np.testing.assert_allclose(out.loc[expected.index, "predict_proba"], expected, atol=1e-9)


def test_woe_expr_matches_optbinning(spark, splits):
    from train_model.spark.binning import woe_transform_spark

    train, _, _ = splits
    b = fit_optbinning(train, ["recharge_amt_l1m"], "GB")
    sdf = spark.createDataFrame(train[["msisdn", "recharge_amt_l1m"]])
    got = woe_transform_spark(sdf, b).toPandas().set_index("msisdn")["recharge_amt_l1m"]
    exp = b["recharge_amt_l1m"].transform(train["recharge_amt_l1m"].to_numpy(), metric="woe",
                                          metric_missing="empirical")
    np.testing.assert_allclose(got.loc[train["msisdn"]].to_numpy(), exp, atol=1e-6)
