"""sklearn-compatible column transformers working on pandas DataFrames.

Every transformer:
* takes ``columns`` (explicit list) or, for numeric ones, ``exclude_columns`` (use all others);
* stores fitted state in attributes ending with ``_`` (so ``sklearn.base.clone`` works);
* returns a new DataFrame and never modifies its input.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


class _ColumnMixin(BaseEstimator, TransformerMixin):
    columns: Sequence[str] | None
    exclude_columns: Sequence[str] | None = None

    def _target_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in X.columns]
        excluded = set(self.exclude_columns or [])
        return [c for c in X.columns if c not in excluded]

    def _cat_columns(self, X: pd.DataFrame) -> list[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in X.columns]
        return X.select_dtypes(include=["object", "category"]).columns.tolist()


class _PerColumnNumeric(_ColumnMixin):
    """Fit one 1-D sklearn transformer per numeric column (NaN kept as NaN)."""

    def _make(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _min_points(self) -> int:
        return 2

    def fit(self, X: pd.DataFrame, y=None):
        self.transformers_ = {}
        for col in self._target_columns(X):
            values = pd.to_numeric(X[col], errors="coerce").dropna()
            if len(values) >= self._min_points():
                self.transformers_[col] = self._make(values).fit(values.to_numpy().reshape(-1, 1))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, tr in self.transformers_.items():
            if col not in X.columns:
                continue
            values = pd.to_numeric(X[col], errors="coerce").astype(float)
            mask = values.notna()
            if mask.any():
                values.loc[mask] = tr.transform(values[mask].to_numpy().reshape(-1, 1)).ravel()
            X[col] = values
        return X


class LogTransformer(_ColumnMixin):
    """``log1p`` on non-negative values of the selected numeric columns (negatives left as-is)."""

    def __init__(self, columns=None, exclude_columns=None):
        self.columns = columns
        self.exclude_columns = exclude_columns

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_columns_ = self._target_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.fitted_columns_:
            if col in X.columns:
                v = pd.to_numeric(X[col], errors="coerce")
                X[col] = np.where(v >= 0, np.log1p(v.clip(lower=0)), v)
        return X


class PowerTransformerWrapper(_PerColumnNumeric):
    """Yeo-Johnson / Box-Cox power transform per column."""

    def __init__(self, columns=None, exclude_columns=None, method="yeo-johnson", standardize=True):
        self.columns = columns
        self.exclude_columns = exclude_columns
        self.method = method
        self.standardize = standardize

    def _make(self, values):
        return PowerTransformer(method=self.method, standardize=self.standardize)


class QuantileBinner(_PerColumnNumeric):
    """Map each column to its quantile (uniform or normal output distribution)."""

    def __init__(self, columns=None, exclude_columns=None, n_quantiles=10, output_distribution="uniform"):
        self.columns = columns
        self.exclude_columns = exclude_columns
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def _min_points(self) -> int:
        return self.n_quantiles + 1

    def _make(self, values):
        n_q = int(min(self.n_quantiles, values.nunique()))
        return QuantileTransformer(n_quantiles=max(n_q, 2), output_distribution=self.output_distribution)


class ScalerWrapper(_PerColumnNumeric):
    """Standard / MinMax / Robust scaling per column."""

    _SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

    def __init__(self, scaler_type="standard", columns=None, exclude_columns=None):
        self.scaler_type = scaler_type
        self.columns = columns
        self.exclude_columns = exclude_columns

    def _make(self, values):
        if self.scaler_type not in self._SCALERS:
            raise ValueError(f"scaler_type must be one of {list(self._SCALERS)}, got {self.scaler_type!r}")
        return self._SCALERS[self.scaler_type]()


class PolynomialFeatureGenerator(_ColumnMixin):
    """Append polynomial / interaction features of the selected numeric columns.

    Returns the original frame plus new columns (original columns of degree 1 are not duplicated).
    NaN are filled with the training median before expansion.
    """

    def __init__(self, columns=None, exclude_columns=None, degree=2, interaction_only=False):
        self.columns = columns
        self.exclude_columns = exclude_columns
        self.degree = degree
        self.interaction_only = interaction_only

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_columns_ = self._target_columns(X)
        num = X[self.fitted_columns_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = num.median()
        self.poly_ = PolynomialFeatures(
            degree=self.degree, interaction_only=self.interaction_only, include_bias=False
        ).fit(num.fillna(self.medians_))
        names = self.poly_.get_feature_names_out(self.fitted_columns_)
        self.new_feature_names_ = [n for n in names if n not in self.fitted_columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.fitted_columns_].apply(pd.to_numeric, errors="coerce").fillna(self.medians_)
        expanded = pd.DataFrame(
            self.poly_.transform(num),
            columns=self.poly_.get_feature_names_out(self.fitted_columns_),
            index=X.index,
        )
        return pd.concat([X.copy(), expanded[self.new_feature_names_]], axis=1)


class LabelEncoderWrapper(_ColumnMixin):
    """Ordinal-encode categorical columns; unseen categories become ``-1``."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.mappings_ = {
            col: {v: i for i, v in enumerate(sorted(X[col].astype(str).unique()))}
            for col in self._cat_columns(X)
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
        return X


class OneHotEncoderWrapper(_ColumnMixin):
    """One-hot encode categorical columns (unknown categories -> all zeros)."""

    def __init__(self, columns=None, drop="first"):
        self.columns = columns
        self.drop = drop

    def fit(self, X: pd.DataFrame, y=None):
        self.fitted_columns_ = self._cat_columns(X)
        self.encoder_ = None
        if self.fitted_columns_:
            self.encoder_ = OneHotEncoder(drop=self.drop, sparse_output=False, handle_unknown="ignore")
            self.encoder_.fit(X[self.fitted_columns_].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None:
            return X.copy()
        encoded = pd.DataFrame(
            self.encoder_.transform(X[self.fitted_columns_].astype(str)),
            columns=self.encoder_.get_feature_names_out(self.fitted_columns_),
            index=X.index,
        )
        return pd.concat([X.drop(columns=self.fitted_columns_), encoded], axis=1)


class FrequencyEncoder(_ColumnMixin):
    """Replace each category with its relative frequency in training data (unseen -> 0)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.freq_maps_ = {c: X[c].value_counts(normalize=True).to_dict() for c in self._cat_columns(X)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, freq in self.freq_maps_.items():
            if col in X.columns:
                X[col] = X[col].map(freq).fillna(0.0).astype(float)
        return X


class TargetEncoder(_ColumnMixin):
    """Smoothed mean-target encoding (unseen categories -> global mean)."""

    def __init__(self, columns=None, smooth=1.0):
        self.columns = columns
        self.smooth = smooth

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_mean_ = float(y.mean())
        self.target_means_ = {}
        for col in self._cat_columns(X):
            stats = y.groupby(X[col]).agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + self.global_mean_ * self.smooth) / (
                stats["count"] + self.smooth
            )
            self.target_means_[col] = smooth.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, means in self.target_means_.items():
            if col in X.columns:
                X[col] = X[col].map(means).fillna(self.global_mean_).astype(float)
        return X


class _StatImputer(_ColumnMixin):
    _numeric = True

    def _stat(self, s: pd.Series):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is not None:
            cols = [c for c in self.columns if c in X.columns]
        elif self._numeric:
            cols = X.select_dtypes(include="number").columns.tolist()
        else:
            cols = X.columns.tolist()
        self.fill_values_ = {c: self._stat(X[c]) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        present = {c: v for c, v in self.fill_values_.items() if c in X.columns}
        return X.fillna(present)


class MeanImputer(_StatImputer):
    def __init__(self, columns=None):
        self.columns = columns

    def _stat(self, s):
        return s.mean()


class MedianImputer(_StatImputer):
    def __init__(self, columns=None):
        self.columns = columns

    def _stat(self, s):
        return s.median()


class ModeImputer(_StatImputer):
    _numeric = False

    def __init__(self, columns=None):
        self.columns = columns

    def _stat(self, s):
        mode = s.mode(dropna=True)
        return mode.iloc[0] if len(mode) else np.nan


class ConstantImputer(_StatImputer):
    _numeric = False

    def __init__(self, columns=None, fill_value="missing"):
        self.columns = columns
        self.fill_value = fill_value

    def _stat(self, s):
        return self.fill_value
