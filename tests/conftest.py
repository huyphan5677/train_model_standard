from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path() -> Path:
    return ROOT / "configs" / "model_config.yaml"


@pytest.fixture(scope="session")
def synthetic_df():
    from train_model.data import make_synthetic_dataset

    return make_synthetic_dataset(n_rows=3000, n_months=8)


@pytest.fixture(scope="session")
def splits(synthetic_df):
    """(train, val, test) with categorical columns already normalised."""
    from train_model.data import time_split
    from train_model.preprocessing import fill_missing

    df = fill_missing(synthetic_df, cat_cols=CAT_COLS)
    split_date = sorted(df["date"].unique())[-2]
    return time_split(df, "date", split_date, val_size=0.25, seed=0, target_col="GB")


CAT_COLS = ["device_manu", "sub_type", "demographic_province_l1m"]
ID_COLS = ["msisdn", "date", "client", "GB"]
