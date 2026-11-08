"""Select features by a fitted model's ``feature_importances_`` (top-k, threshold or cumulative share)."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone


def select_by_importance(
    X: pd.DataFrame,
    y,
    estimator,
    top_k: int | None = None,
    threshold: float | None = None,
    accumulate: float | None = None,
    fit_kwargs: dict | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Fit ``estimator`` and pick features by normalized importance (sums to 1).

    Priority: ``accumulate`` > ``top_k`` > ``threshold``.

    * ``accumulate=0.95``: keep features (most important first) until cumulative importance reaches 95 %
      (the feature that crosses the line is kept).
    * ``top_k=30``: keep the 30 most important.
    * ``threshold=0.01``: keep features with normalized importance >= 1 %.

    Returns ``(selected_features, importance_df[feature, importance, cumulative])``.
    """
    if accumulate is None and top_k is None and threshold is None:
        raise ValueError("Specify one of: accumulate, top_k, threshold")

    estimator.fit(X, y, **(fit_kwargs or {}))
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Estimator must expose feature_importances_")

    imp = pd.DataFrame({"feature": X.columns, "importance": estimator.feature_importances_})
    total = imp["importance"].sum()
    imp["importance"] = imp["importance"] / total if total > 0 else 1.0 / len(imp)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["cumulative"] = imp["importance"].cumsum()

    if accumulate is not None:
        before = imp["cumulative"] - imp["importance"]
        selected = imp.loc[before < accumulate, "feature"].tolist()
    elif top_k is not None:
        selected = imp["feature"].head(top_k).tolist()
    else:
        selected = imp.loc[imp["importance"] >= threshold, "feature"].tolist()
    return selected, imp


class ImportanceSelector(BaseEstimator, TransformerMixin):
    """Transformer wrapper of :func:`select_by_importance`.

    Default estimator: ``CatBoostClassifier(iterations=300)`` with object columns as categorical.
    """

    def __init__(self, estimator=None, top_k=None, threshold=None, accumulate=None):
        self.estimator = estimator
        self.top_k = top_k
        self.threshold = threshold
        self.accumulate = accumulate

    def fit(self, X: pd.DataFrame, y):
        if self.estimator is None:
            from catboost import CatBoostClassifier

            cats = X.select_dtypes(exclude="number").columns.tolist()
            est = CatBoostClassifier(iterations=300, verbose=False, random_seed=42, cat_features=cats)
        else:
            est = clone(self.estimator)
        self.selected_features_, self.importance_ = select_by_importance(
            X, y, est, top_k=self.top_k, threshold=self.threshold, accumulate=self.accumulate
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()
