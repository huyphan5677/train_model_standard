"""Spark-module tests that run WITHOUT pyspark (pure-Python parts + import safety)."""

import importlib

import numpy as np
import pytest

from train_model.feature_selection import fit_optbinning
from train_model.spark.binning import apply_woe_bins, woe_bins, woe_case_when_expr
from train_model.spark.session import s3a_conf_from_env


@pytest.mark.parametrize(
    "module",
    ["train_model.spark.session", "train_model.spark.predict", "train_model.spark.binning",
     "train_model.spark.scoring", "train_model.spark.shap"],
)
def test_spark_modules_import_without_pyspark(module):
    importlib.import_module(module)


def test_core_package_does_not_import_pyspark():
    import subprocess
    import sys

    code = (
        "import sys, train_model.data, train_model.preprocessing, train_model.feature_selection, "
        "train_model.modeling, train_model.evaluation, train_model.scoring; "
        "print('pyspark' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.fixture(scope="module")
def binnings(splits):
    train, _, _ = splits
    return train, fit_optbinning(train, ["recharge_amt_l1m", "voice_out_min_l1m", "device_manu"], "GB",
                                 cat_cols=["device_manu"])


@pytest.mark.parametrize("feature", ["recharge_amt_l1m", "voice_out_min_l1m", "device_manu"])
def test_woe_bins_match_optbinning_transform(binnings, feature):
    train, b = binnings
    optb = b[feature]
    values = train[feature].to_numpy(dtype=object if optb.dtype == "categorical" else float)
    expected = optb.transform(values, metric="woe", metric_missing="empirical")
    got = apply_woe_bins(values, woe_bins(optb))
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_woe_case_when_expr_shape(binnings):
    _, b = binnings
    num = woe_case_when_expr("recharge_amt_l1m", woe_bins(b["recharge_amt_l1m"]))
    cat = woe_case_when_expr("device_manu", woe_bins(b["device_manu"]))
    assert num.startswith("CASE WHEN `recharge_amt_l1m` IS NULL") and num.endswith("ELSE 0.0 END")
    assert "inf" not in num
    assert "IN ('" in cat


def test_s3a_conf_from_env(monkeypatch):
    for var in ("S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert s3a_conf_from_env() == {}
    monkeypatch.setenv("S3_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "k")
    monkeypatch.setenv("S3_SECRET_KEY", "s")
    conf = s3a_conf_from_env()
    assert conf["spark.hadoop.fs.s3a.endpoint"] == "http://minio:9000"
    assert conf["spark.hadoop.fs.s3a.access.key"] == "k"
