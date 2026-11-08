import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from train_model.preprocessing import (
    CapCategoricalOutlier,
    ConstantImputer,
    FrequencyEncoder,
    LabelEncoderWrapper,
    LogTransformer,
    MeanImputer,
    MedianImputer,
    ModeImputer,
    OneHotEncoderWrapper,
    PolynomialFeatureGenerator,
    PowerTransformerWrapper,
    QuantileBinner,
    ScalerWrapper,
    TargetEncoder,
    cat_missing_step,
    cat_outlier_step,
    fill_missing,
    num_missing_step,
    num_outlier_step,
)


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "a": rng.lognormal(size=n),
            "b": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
            "c": rng.choice(["x", "y", "z"], size=n).astype(object),
        }
    )


def test_fill_missing_normalizes_cats():
    df = pd.DataFrame({"dev": [" Apple", None, "", "SAMSUNG", "nan"], "v": [1.0, np.nan, 2.0, 3.0, 4.0]})
    before = df.copy()
    out = fill_missing(df, cat_cols=["dev"])
    assert out["dev"].tolist() == ["apple", "unknown", "unknown", "samsung", "unknown"]
    assert out["v"].isna().sum() == 1  # num_fill=None keeps NaN
    pd.testing.assert_frame_equal(df, before)


def test_fill_missing_num_fill():
    df = pd.DataFrame({"dev": ["a", None], "v": [1.0, np.nan], "w": [np.nan, 2]})
    out = fill_missing(df, cat_cols=["dev"], cat_fill="other", num_fill=0)
    assert out["dev"].tolist() == ["a", "other"]
    assert out["v"].tolist() == [1.0, 0.0]
    assert out["w"].tolist() == [0.0, 2.0]


def test_fill_missing_ignores_absent_columns():
    df = pd.DataFrame({"v": [1.0]})
    assert fill_missing(df, cat_cols=["not_here"]).equals(df)


NUM = ["a", "b"]
TRANSFORMERS = [
    LogTransformer(columns=NUM),
    PowerTransformerWrapper(columns=NUM),
    QuantileBinner(columns=NUM, n_quantiles=5),
    ScalerWrapper(scaler_type="robust", columns=NUM),
    ScalerWrapper(scaler_type="minmax", columns=NUM),
    LabelEncoderWrapper(columns=["c"]),
    OneHotEncoderWrapper(columns=["c"]),
    FrequencyEncoder(columns=["c"]),
    TargetEncoder(columns=["c"]),
    MeanImputer(columns=NUM),
    MedianImputer(columns=NUM),
    ModeImputer(columns=["c"]),
    ConstantImputer(columns=["c"], fill_value="missing"),
    PolynomialFeatureGenerator(columns=NUM, degree=2),
]


@pytest.mark.parametrize("transformer", TRANSFORMERS, ids=lambda t: type(t).__name__)
def test_transformers_clone_and_fit(transformer, small_df):
    y = pd.Series(np.arange(len(small_df)) % 2)
    before = small_df.copy()
    fitted = clone(transformer).fit(small_df, y)
    out = fitted.transform(small_df)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(small_df.index)
    pd.testing.assert_frame_equal(small_df, before)


def test_imputers_fill_nan(small_df):
    out = MedianImputer(columns=["b"]).fit(small_df).transform(small_df)
    assert out["b"].isna().sum() == 0


def test_encoders_handle_nan_categories():
    df = pd.DataFrame({"c": ["x", None, "y", np.nan]})
    assert LabelEncoderWrapper(columns=["c"]).fit(df).transform(df)["c"].nunique() == 3
    assert OneHotEncoderWrapper(columns=["c"], drop=None).fit(df).transform(df).shape[1] == 3


def test_label_encoder_unseen_category(small_df):
    enc = LabelEncoderWrapper(columns=["c"]).fit(small_df)
    out = enc.transform(pd.DataFrame({"c": ["x", "never_seen"]}))
    assert out["c"].iloc[1] == -1


def test_cat_missing_step_arbitrary():
    name, imputer = cat_missing_step(["c"], "arbitrary", fill_value="missing")
    df = pd.DataFrame({"c": ["x", None]})
    assert imputer.fit_transform(df)["c"].tolist() == ["x", "missing"]
    assert name == "cat_imputer"


def test_num_missing_and_outlier_steps():
    df = pd.DataFrame({"v": [1.0, 2.0, np.nan, 3.0, 100.0] * 10})
    _, imp = num_missing_step(["v"], "median")
    filled = imp.fit_transform(df)
    assert filled["v"].isna().sum() == 0
    _, win = num_outlier_step("quantiles", ["v"], fold=0.05, tail="both")
    capped = win.fit_transform(filled)
    assert capped["v"].max() <= filled["v"].max()
    assert num_outlier_step("unknown", ["v"]) is None


def test_cat_outlier_step():
    _, tr = cat_outlier_step({"c": {"rare": "other"}})
    assert isinstance(tr, CapCategoricalOutlier)
    assert tr.fit_transform(pd.DataFrame({"c": ["rare", "x"]}))["c"].tolist() == ["other", "x"]
    assert cat_outlier_step({}) is None
