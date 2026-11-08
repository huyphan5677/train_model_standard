"""SHAP-based feature importance for tree models (CatBoost / sklearn trees)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    try:
        from catboost import CatBoostClassifier, Pool

        if isinstance(model, CatBoostClassifier):
            cats = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
            pool = Pool(X, cat_features=cats)
            values = model.get_feature_importance(pool, type="ShapValues")
            return values[:, :-1]  # last column is the expected value
    except ImportError:  # pragma: no cover
        pass
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def sample_rows(X: pd.DataFrame, max_samples: int, seed: int = 42) -> pd.DataFrame:
    return X.sample(n=max_samples, random_state=seed) if len(X) > max_samples else X


def shap_importance(model, X: pd.DataFrame, max_samples: int = 5000) -> pd.DataFrame:
    """Mean |SHAP| per feature on (a sample of) ``X``. Returns ``DataFrame[feature, mean_abs_shap]`` sorted desc."""
    Xs = sample_rows(X, max_samples)
    values = _shap_values(model, Xs)
    out = pd.DataFrame({"feature": Xs.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def shap_group_importance(shap_df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """Aggregate :func:`shap_importance` output by feature-name prefix (``usage_x`` -> ``usage``)."""
    df = shap_df.copy()
    df["group"] = df["feature"].astype(str).str.split(sep).str[0]
    out = df.groupby("group")["mean_abs_shap"].agg(["mean", "std", "count", "max", "min", "sum"]).reset_index()
    return out.sort_values("sum", ascending=False).reset_index(drop=True)
