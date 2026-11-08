"""Pipeline step builders for missing-value imputation and outlier capping (feature-engine based).

Each ``*_step`` function returns a ``(name, transformer)`` tuple ready to put in
``sklearn.pipeline.Pipeline(steps=[...])`` (or ``None`` when the step is not applicable).
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

_OUTLIER_METHODS = ("gaussian", "iqr", "mad", "quantiles")


def num_missing_step(numerical: list[str], method: str, arbitrary_number: float = 0):
    """``method`` in {'mean', 'median', 'arbitrary'}."""
    from feature_engine.imputation import ArbitraryNumberImputer, MeanMedianImputer

    if method == "arbitrary":
        return "num_imputer", ArbitraryNumberImputer(arbitrary_number=arbitrary_number, variables=numerical)
    if method in ("mean", "median"):
        return "num_imputer", MeanMedianImputer(imputation_method=method, variables=numerical)
    raise ValueError(f"method must be 'mean', 'median' or 'arbitrary', got {method!r}")


def cat_missing_step(categorical: list[str], method: str, fill_value: str = "missing"):
    """``method`` in {'mode', 'arbitrary'}."""
    from feature_engine.imputation import CategoricalImputer

    if method == "mode":
        return "cat_imputer", CategoricalImputer(imputation_method="frequent", variables=categorical)
    if method == "arbitrary":
        return "cat_imputer", CategoricalImputer(
            imputation_method="missing", fill_value=fill_value, variables=categorical
        )
    raise ValueError(f"method must be 'mode' or 'arbitrary', got {method!r}")


class CapCategoricalOutlier(BaseEstimator, TransformerMixin):
    """Replace rare / invalid categorical values using a mapping.

    ``dict_replace = {"OCCUPATION_TYPE": {"Unknown": "Other"}, "ORG_TYPE": {"XNA": None}}``
    """

    def __init__(self, dict_replace=None):
        self.dict_replace = dict_replace

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature, mapping in (self.dict_replace or {}).items():
            if feature in X.columns:
                X[feature] = X[feature].replace(mapping)
        return X


def num_outlier_step(method: str, numerical: list[str], fold: float = 3, tail: str = "right"):
    """Winsorizer step. ``method`` in {'gaussian', 'iqr', 'mad', 'quantiles'}; otherwise ``None``.

    ``fold``: multiplier of std / IQR / MAD, or the tail percentile for 'quantiles' (e.g. 0.05).
    """
    from feature_engine.outliers import Winsorizer

    if method not in _OUTLIER_METHODS:
        return None
    return "num_outlier", Winsorizer(capping_method=method, tail=tail, fold=fold, variables=numerical)


def cat_outlier_step(dict_replace: dict | None):
    if not isinstance(dict_replace, dict) or not dict_replace:
        return None
    return "cat_outlier", CapCategoricalOutlier(dict_replace=dict_replace)
