"""Synthetic telco-like dataset so the training notebook runs anywhere without real data.

The data deliberately contains "traps" that feature selection is expected to remove:
a constant column, an exact duplicate, a highly correlated copy, mostly/all-missing columns
and pure-noise columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INFORMATIVE_FEATURES: tuple[str, ...] = (
    "usage_data_mb_l1m",
    "usage_data_mb_l3m",
    "recharge_amt_l1m",
    "recharge_cnt_l3m",
    "voice_out_min_l1m",
    "sms_out_cnt_l1m",
    "demographic_age_l1m",
)

_DEVICE = ["samsung", "apple", "oppo", "xiaomi", "vivo", "other brands"]
_SUB_TYPE = ["prepaid", "postpaid"]
_PROVINCE = [f"p{i:02d}" for i in range(1, 11)]


def _month_starts(start: str, n_months: int) -> list[int]:
    first = pd.Timestamp(start).to_period("M").to_timestamp()
    return [int(d.strftime("%Y%m%d")) for d in pd.date_range(first, periods=n_months, freq="MS")]


def _with_missing(values: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    out = values.astype(object)
    out[rng.random(len(out)) < rate] = None
    return out


def make_synthetic_dataset(
    n_rows: int = 6000,
    n_months: int = 12,
    start: str = "20240801",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible binary-classification dataset with telco-style feature names.

    Columns: ``msisdn``, ``date`` (int yyyymmdd, first day of month), ``client`` (A/B), label ``GB``
    (1 = good), informative numeric features (:data:`INFORMATIVE_FEATURES`), categorical
    ``device_manu`` / ``sub_type`` / ``demographic_province_l1m`` (with NaN and empty strings),
    ``noise_1..noise_5`` and the trap columns ``const_col``, ``dup_of_usage``, ``corr_of_recharge``,
    ``mostly_missing``, ``all_missing``.
    """
    rng = np.random.default_rng(seed)
    months = _month_starts(start, n_months)

    df = pd.DataFrame(
        {
            "msisdn": [f"856{n:08d}" for n in rng.choice(10**8, size=n_rows, replace=False)],
            "date": rng.choice(months, size=n_rows),
            "client": rng.choice(["A", "B"], size=n_rows, p=[0.6, 0.4]),
        }
    )

    usage_l1m = rng.lognormal(mean=6.0, sigma=1.0, size=n_rows)
    df["usage_data_mb_l1m"] = usage_l1m
    df["usage_data_mb_l3m"] = usage_l1m + 2 * rng.lognormal(mean=6.0, sigma=1.0, size=n_rows)
    df["recharge_amt_l1m"] = rng.gamma(shape=2.0, scale=50_000, size=n_rows)
    df["recharge_cnt_l3m"] = rng.poisson(lam=6, size=n_rows).astype(float)
    df["voice_out_min_l1m"] = rng.gamma(shape=1.5, scale=80, size=n_rows)
    df["sms_out_cnt_l1m"] = rng.poisson(lam=15, size=n_rows).astype(float)
    df["demographic_age_l1m"] = rng.integers(18, 70, size=n_rows).astype(float)

    df["device_manu"] = _with_missing(rng.choice(_DEVICE, size=n_rows), 0.05, rng)
    df["sub_type"] = rng.choice(_SUB_TYPE, size=n_rows, p=[0.8, 0.2]).astype(object)
    df.loc[rng.random(n_rows) < 0.02, "sub_type"] = ""
    df["demographic_province_l1m"] = _with_missing(rng.choice(_PROVINCE, size=n_rows), 0.03, rng)

    # --- label: logistic model over standardized informative features ---
    def z(col: str) -> np.ndarray:
        x = np.log1p(df[col].to_numpy(dtype=float))
        return (x - x.mean()) / x.std()

    logit = (
        -1.25
        + 0.9 * z("recharge_amt_l1m")
        + 0.6 * z("usage_data_mb_l1m")
        + 0.4 * z("usage_data_mb_l3m")
        + 0.5 * z("recharge_cnt_l3m")
        + 0.35 * z("voice_out_min_l1m")
        + 0.25 * z("sms_out_cnt_l1m")
        + 0.3 * (df["demographic_age_l1m"].to_numpy() - 44) / 15
        + np.where(df["sub_type"].to_numpy() == "postpaid", 1.2, 0.0)
        + np.where(df["device_manu"].to_numpy() == "apple", 1.0, 0.0)
        + np.where(df["device_manu"].to_numpy() == "other brands", -0.6, 0.0)
    )
    df["GB"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    # --- noise and traps ---
    for i in range(1, 6):
        df[f"noise_{i}"] = rng.normal(size=n_rows)
    df["const_col"] = 1.0
    df["dup_of_usage"] = df["usage_data_mb_l1m"]
    df["corr_of_recharge"] = df["recharge_amt_l1m"] * 1.01 + rng.normal(0, 100, n_rows)
    mostly = rng.normal(size=n_rows)
    mostly[rng.random(n_rows) < 0.98] = np.nan
    df["mostly_missing"] = mostly
    df["all_missing"] = np.nan

    # a few missing values inside informative features
    for col in ("voice_out_min_l1m", "sms_out_cnt_l1m"):
        df.loc[rng.random(n_rows) < 0.03, col] = np.nan

    return df.sort_values(["date", "msisdn"]).reset_index(drop=True)
