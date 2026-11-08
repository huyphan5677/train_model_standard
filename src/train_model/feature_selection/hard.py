"""Hard feature selection: rule-based removal of useless / redundant features.

Steps (in order), each recorded in ``report_`` with the reason value:

1. ``missing``   — missing ratio > ``missing_threshold``
2. ``constant``  — one value covers >= ``quasi_constant_tol`` of rows (NaN counts as a value)
3. ``duplicate`` — exact duplicate of an earlier column
4. ``low_iv``    — Information Value < ``iv_threshold`` (or binning failed)
5. ``high_corr`` — numeric pair with |Pearson corr| > ``corr_threshold``: drop the lower-IV member

``transform`` returns the selected columns with their original values (use
:class:`~train_model.feature_selection.binning.WoETransformer` separately if WoE values are wanted).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from train_model.feature_selection.binning import binning_iv, fit_optbinning


def _constant_features(X: pd.DataFrame, tol: float) -> dict[str, float]:
    out = {}
    for col in X.columns:
        top_share = X[col].value_counts(normalize=True, dropna=False).iloc[0] if len(X) else 1.0
        if top_share >= tol:
            out[col] = float(top_share)
    return out


def _duplicate_features(X: pd.DataFrame) -> dict[str, str]:
    """Map duplicate column -> the earlier column it duplicates."""
    seen: dict[int, list[str]] = {}
    dups: dict[str, str] = {}
    for col in X.columns:
        s = X[col]
        key = int(pd.util.hash_pandas_object(s, index=False).sum())
        for prev in seen.get(key, []):
            if s.equals(X[prev]):
                dups[col] = prev
                break
        else:
            seen.setdefault(key, []).append(col)
    return dups


class HardFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        missing_threshold: float = 0.95,
        quasi_constant_tol: float = 0.98,
        iv_threshold: float = 0.02,
        corr_threshold: float = 0.9,
        cat_cols: Sequence[str] = (),
        max_n_bins: int = 10,
    ):
        self.missing_threshold = missing_threshold
        self.quasi_constant_tol = quasi_constant_tol
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.cat_cols = cat_cols
        self.max_n_bins = max_n_bins

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y), index=X.index)
        report: list[dict] = []
        remaining = list(X.columns)

        def drop(features: dict, step: str) -> None:
            for feat, value in features.items():
                report.append({"feature": feat, "step": step, "value": value})
                remaining.remove(feat)

        # 1. missing ratio
        miss = X[remaining].isna().mean()
        drop(miss[miss > self.missing_threshold].to_dict(), "missing")

        # 2. constant / quasi-constant
        drop(_constant_features(X[remaining], self.quasi_constant_tol), "constant")

        # 3. exact duplicates
        drop(_duplicate_features(X[remaining]), "duplicate")

        # 4. information value
        binnings = fit_optbinning(X, remaining, y, self.cat_cols, self.max_n_bins)
        iv = pd.Series({f: binning_iv(binnings[f]) if f in binnings else np.nan for f in remaining}, dtype=float)
        low = iv[(iv < self.iv_threshold) | iv.isna()]
        drop(low.to_dict(), "low_iv")

        # 5. high correlation among numeric survivors (keep the higher-IV feature)
        num = [c for c in remaining if c not in self.cat_cols and pd.api.types.is_numeric_dtype(X[c])]
        if len(num) > 1:
            corr = X[num].corr().abs()
            pairs = [
                (corr.iloc[i, j], num[i], num[j])
                for i in range(len(num))
                for j in range(i + 1, len(num))
                if corr.iloc[i, j] > self.corr_threshold
            ]
            removed: dict[str, float] = {}
            for value, a, b in sorted(pairs, key=lambda t: -t[0]):
                if a in removed or b in removed:
                    continue
                loser = a if iv[a] < iv[b] else b
                removed[loser] = float(value)
            drop(removed, "high_corr")

        self.iv_ = iv.rename("iv")
        self.binnings_ = binnings
        self.selected_features_ = remaining
        self.report_ = pd.DataFrame(report, columns=["feature", "step", "value"])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)
