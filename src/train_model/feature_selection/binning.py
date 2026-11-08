"""Optimal binning (optbinning): WoE transform and Information Value — the single IV/WoE implementation."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from optbinning import OptimalBinning
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def _prepare(series: pd.Series, is_cat: bool) -> np.ndarray:
    if is_cat:
        obj = series.astype(object)
        return obj.where(obj.isna(), obj.map(str)).to_numpy()
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _is_categorical(series: pd.Series, name: str, cat_cols: Sequence[str]) -> bool:
    return name in cat_cols or not pd.api.types.is_numeric_dtype(series)


def fit_one_binning(
    x: pd.Series, y: pd.Series | np.ndarray, name: str, is_cat: bool, max_n_bins: int = 10
) -> OptimalBinning:
    """Fit a single :class:`OptimalBinning` (raises on failure)."""
    optb = OptimalBinning(
        name=name,
        dtype="categorical" if is_cat else "numerical",
        solver="cp",
        max_n_bins=max_n_bins,
    )
    optb.fit(_prepare(x, is_cat), np.asarray(y))
    return optb


def binning_iv(optb: OptimalBinning) -> float:
    """Total Information Value of a fitted binning."""
    table = optb.binning_table.build()
    return float(table.loc["Totals", "IV"])


def fit_optbinning(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str | pd.Series,
    cat_cols: Sequence[str] = (),
    max_n_bins: int = 10,
) -> dict[str, OptimalBinning]:
    """Fit one OptimalBinning per feature. Features that fail are logged and skipped.

    ``target`` is a column name of ``df`` or a Series aligned with ``df``.
    """
    y = df[target] if isinstance(target, str) else target
    result: dict[str, OptimalBinning] = {}
    for feat in features:
        try:
            is_cat = _is_categorical(df[feat], feat, cat_cols)
            result[feat] = fit_one_binning(df[feat], y, feat, is_cat, max_n_bins)
        except Exception as exc:  # noqa: BLE001 - optbinning raises many types
            logger.warning("optbinning failed for %s: %s", feat, exc)
    return result


def compute_iv(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str | pd.Series,
    cat_cols: Sequence[str] = (),
    max_n_bins: int = 10,
) -> pd.DataFrame:
    """Information Value per feature, sorted descending. Features whose binning fails get ``NaN``."""
    binnings = fit_optbinning(df, features, target, cat_cols, max_n_bins)
    rows = [{"feature": f, "iv": binning_iv(binnings[f]) if f in binnings else np.nan} for f in features]
    return pd.DataFrame(rows).sort_values("iv", ascending=False, na_position="last").reset_index(drop=True)


class WoETransformer(BaseEstimator, TransformerMixin):
    """Replace feature values by their Weight of Evidence (fit with optbinning).

    ``features=None`` uses every column of X. Missing values get the empirical WoE of the Missing bin.
    """

    def __init__(self, features: Sequence[str] | None = None, cat_cols: Sequence[str] = (), max_n_bins: int = 10):
        self.features = features
        self.cat_cols = cat_cols
        self.max_n_bins = max_n_bins

    def fit(self, X: pd.DataFrame, y):
        feats = list(self.features) if self.features is not None else list(X.columns)
        self.binnings_ = fit_optbinning(X, feats, pd.Series(np.asarray(y), index=X.index), self.cat_cols, self.max_n_bins)
        self.iv_ = pd.Series({f: binning_iv(b) for f, b in self.binnings_.items()}, name="iv")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feat, optb in self.binnings_.items():
            if feat in X.columns:
                is_cat = optb.dtype == "categorical"
                X[feat] = optb.transform(_prepare(X[feat], is_cat), metric="woe", metric_missing="empirical")
        return X
