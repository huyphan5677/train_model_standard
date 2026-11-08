"""Missing-value handling shared by training and prediction."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

_NULL_TOKENS = {"", "nan", "none", "null"}


def normalize_category(series: pd.Series, fill: str = "unknown") -> pd.Series:
    """str -> strip -> lower; NaN / empty / 'nan' / 'none' / 'null' -> ``fill``."""
    out = series.astype("string").str.strip().str.lower()
    out = out.mask(out.isna() | out.isin(_NULL_TOKENS), fill)
    return out.astype(object)


def fill_missing(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    cat_fill: str = "unknown",
    num_fill: float | None = None,
) -> pd.DataFrame:
    """Return a copy with categorical columns normalized and (optionally) numeric NaN filled.

    * ``cat_cols`` (only those present) are cast to str, stripped, lower-cased; empty / null -> ``cat_fill``.
    * Numeric columns are filled with ``num_fill`` when it is not ``None``; otherwise NaN is kept
      (CatBoost handles missing numeric values natively).
    """
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            out[col] = normalize_category(out[col], cat_fill)
    if num_fill is not None:
        num_cols = out.select_dtypes(include="number").columns
        out[num_cols] = out[num_cols].fillna(num_fill)
    return out
