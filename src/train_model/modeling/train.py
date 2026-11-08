"""CatBoost training and split-wise evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from catboost import CatBoostClassifier

from train_model.evaluation.metrics import auc_gini
from train_model.preprocessing.fill import to_category_str

DEFAULT_PARAMS = {
    "iterations": 500,
    "depth": 4,
    "learning_rate": 0.05,
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
}


def prepare_features(df: pd.DataFrame, features: Sequence[str], cat_features: Sequence[str]) -> pd.DataFrame:
    """Select ``features`` (in order) and cast categorical ones to str (CatBoost requirement). Returns a copy."""
    X = df[list(features)].copy()
    for c in cat_features:
        if c in X.columns:
            X[c] = to_category_str(X[c])
    return X


def train_catboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: Sequence[str],
    target: str,
    cat_features: Sequence[str] = (),
    params: dict | None = None,
    seed: int = 42,
    verbose: int | bool = False,
) -> CatBoostClassifier:
    """Fit a CatBoostClassifier on ``train`` with ``val`` as eval set (best iteration kept).

    ``params`` override :data:`DEFAULT_PARAMS`. Numeric NaN are left to CatBoost; categorical values are
    cast to str (so NaN becomes the category ``"nan"`` — normalize with ``fill_missing`` first).
    """
    cats = [c for c in cat_features if c in features]
    model = CatBoostClassifier(
        **{**DEFAULT_PARAMS, **(params or {})},
        cat_features=cats,
        random_seed=seed,
        verbose=verbose,
    )
    model.fit(
        prepare_features(train, features, cats),
        train[target],
        eval_set=(prepare_features(val, features, cats), val[target]),
        use_best_model=True,
    )
    return model


def evaluate_splits(
    model: CatBoostClassifier,
    splits: dict[str, pd.DataFrame],
    features: Sequence[str],
    target: str,
) -> pd.DataFrame:
    """AUC / Gini on each named split. Returns ``DataFrame[split, n, auc, gini]`` (inputs untouched)."""
    cats = [model.feature_names_[i] for i in model.get_cat_feature_indices()]
    rows = []
    for name, df in splits.items():
        proba = model.predict_proba(prepare_features(df, features, cats))[:, 1]
        rows.append({"split": name, "n": len(df), **auc_gini(df[target], proba)})
    return pd.DataFrame(rows, columns=["split", "n", "auc", "gini"])
