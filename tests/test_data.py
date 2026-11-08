import pandas as pd
import pytest

from train_model.data import (
    INFORMATIVE_FEATURES,
    label_summary,
    make_synthetic_dataset,
    time_split,
)


def test_synthetic_shape_and_traps():
    df = make_synthetic_dataset()
    assert len(df) == 6000
    assert df["const_col"].nunique() == 1
    assert (df["dup_of_usage"] == df["usage_data_mb_l1m"]).all()
    assert df["all_missing"].isna().all()
    assert df["mostly_missing"].isna().mean() >= 0.97
    assert 0.15 < df["GB"].mean() < 0.4
    assert set(INFORMATIVE_FEATURES) <= set(df.columns)
    pd.testing.assert_frame_equal(df, make_synthetic_dataset())


def test_time_split_by_date(synthetic_df):
    before = synthetic_df.copy()
    split = sorted(synthetic_df["date"].unique())[-2]
    train, val, test = time_split(synthetic_df, "date", split, val_size=0.2, seed=1, target_col="GB")
    assert (test["date"] >= split).all()
    assert (train["date"] < split).all() and (val["date"] < split).all()
    assert len(train) + len(val) + len(test) == len(synthetic_df)
    assert abs(len(val) / (len(train) + len(val)) - 0.2) < 0.01
    pd.testing.assert_frame_equal(synthetic_df, before)


def test_time_split_by_group_dict(synthetic_df):
    dates = sorted(synthetic_df["date"].unique())
    split = {"A": dates[-2], "B": dates[-3]}
    train, val, test = time_split(synthetic_df, "date", split, group_col="client")
    for client, d in split.items():
        assert (test.loc[test["client"] == client, "date"] >= d).all()
        dev = pd.concat([train, val])
        assert (dev.loc[dev["client"] == client, "date"] < d).all()


def test_time_split_dict_requires_group(synthetic_df):
    with pytest.raises(ValueError, match="group_col"):
        time_split(synthetic_df, "date", {"A": 20250101})


def test_label_summary(synthetic_df):
    res = label_summary(synthetic_df, "GB", "date", "client")
    assert res["overall"]["rate"].iloc[0] == pytest.approx(synthetic_df["GB"].mean())
    assert len(res["by_month"]) == synthetic_df["date"].nunique()
    assert set(res["by_group"]["client"]) == {"A", "B"}
    assert "by_group" not in label_summary(synthetic_df, "GB", "date")
