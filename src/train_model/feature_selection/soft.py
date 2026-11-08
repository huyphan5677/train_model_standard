"""Soft feature selection: several independent criteria each vote for a feature; keep features
with at least ``min_votes`` votes. Then :func:`forward_selection_gini` adds features one by one in
vote order to find how many are really needed.

Criteria (rank-based ones vote when the feature is in the top ``top_k``):

==================  ==========================================================================
correlation         |Pearson r| between feature and label (train)
permutation         AUC drop on validation when the feature is shuffled (RandomForest)
single_feature      validation AUC of a RandomForest trained on that feature alone
tree_importance     RandomForest impurity importance
tree_vs_random      importance higher than an injected random column (not rank based)
variance            variance > ``variance_threshold`` (not rank based)
mutual_info         in the top ``mi_percentile`` % by mutual information
l1 / l2             kept by ``SelectFromModel(LogisticRegression(penalty))`` on standardized data
rfe                 selected by RFE(DecisionTree) with ``n_features_to_select=top_k``
==================  ==========================================================================
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectPercentile, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from train_model.preprocessing.fill import to_category_str

ALL_CRITERIA: tuple[str, ...] = (
    "correlation",
    "permutation",
    "single_feature",
    "tree_importance",
    "tree_vs_random",
    "variance",
    "mutual_info",
    "l1",
    "l2",
    "rfe",
)

_RANDOM_COL = "__random_feature__"


class _Encoder:
    """Turn a mixed DataFrame into a numeric matrix (ordinal cats, median-filled numerics)."""

    def __init__(self, X: pd.DataFrame, cat_cols: Sequence[str]):
        self.cat_cols = [c for c in X.columns if c in cat_cols or not pd.api.types.is_numeric_dtype(X[c])]
        self.num_cols = [c for c in X.columns if c not in self.cat_cols]
        self.categories = {
            c: {v: i for i, v in enumerate(sorted(to_category_str(X[c]).unique()))} for c in self.cat_cols
        }
        self.medians = X[self.num_cols].median()

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            if c in self.categories:
                out[c] = to_category_str(X[c]).map(self.categories[c]).fillna(-1).astype(float)
            else:
                out[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.medians[c]).astype(float)
        return out.fillna(0.0)


def _top_k_votes(scores: pd.Series, k: int) -> pd.Series:
    ranks = rankdata(-scores.fillna(-np.inf).to_numpy(), method="min")
    return pd.Series(ranks <= k, index=scores.index)


class SoftFeatureSelector:
    def __init__(
        self,
        top_k: int = 15,
        min_votes: int = 3,
        criteria: Sequence[str] = ALL_CRITERIA,
        cat_cols: Sequence[str] = (),
        variance_threshold: float = 0.1,
        mi_percentile: int = 60,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        unknown = [c for c in criteria if c not in ALL_CRITERIA]
        if unknown:
            raise ValueError(f"Unknown criteria {unknown}; valid: {list(ALL_CRITERIA)}")
        self.top_k = top_k
        self.min_votes = min_votes
        self.criteria = tuple(criteria)
        self.cat_cols = tuple(cat_cols)
        self.variance_threshold = variance_threshold
        self.mi_percentile = mi_percentile
        self.n_estimators = n_estimators
        self.random_state = random_state

    # ---------------------------------------------------------------- criteria
    def _rf(self, depth: int, n_estimators: int | None = None) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators or self.n_estimators,
            max_depth=depth,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _correlation(self, X, y, Xv, yv):
        scores = X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]) if col.std() > 0 else 0.0)
        return _top_k_votes(scores, self.top_k)

    def _permutation(self, X, y, Xv, yv):
        model = self._rf(depth=6).fit(X, y)
        res = permutation_importance(
            model, Xv, yv, scoring="roc_auc", n_repeats=3, random_state=self.random_state, n_jobs=-1
        )
        return _top_k_votes(pd.Series(res.importances_mean, index=X.columns), self.top_k)

    def _single_feature(self, X, y, Xv, yv):
        scores = {}
        for col in X.columns:
            model = self._rf(depth=3, n_estimators=min(self.n_estimators, 50)).fit(X[[col]], y)
            scores[col] = roc_auc_score(yv, model.predict_proba(Xv[[col]])[:, 1])
        return _top_k_votes(pd.Series(scores), self.top_k)

    def _forest_with_random(self, X, y) -> pd.Series:
        if not hasattr(self, "_forest_importance"):
            rng = np.random.default_rng(self.random_state)
            Xr = X.assign(**{_RANDOM_COL: rng.permutation(np.asarray(y, dtype=float))})
            model = self._rf(depth=6).fit(Xr, y)
            self._forest_importance = pd.Series(model.feature_importances_, index=Xr.columns)
        return self._forest_importance

    def _tree_importance(self, X, y, Xv, yv):
        imp = self._forest_with_random(X, y).drop(_RANDOM_COL)
        return _top_k_votes(imp, self.top_k)

    def _tree_vs_random(self, X, y, Xv, yv):
        imp = self._forest_with_random(X, y)
        return imp.drop(_RANDOM_COL) > imp[_RANDOM_COL]

    def _variance(self, X, y, Xv, yv):
        return X.var() > self.variance_threshold

    def _mutual_info(self, X, y, Xv, yv):
        discrete = [c in self._encoder.categories for c in X.columns]
        selector = SelectPercentile(
            lambda a, b: mutual_info_classif(a, b, discrete_features=discrete, random_state=self.random_state),
            percentile=self.mi_percentile,
        ).fit(X, y)
        return pd.Series(selector.get_support(), index=X.columns)

    def _regularization(self, X, y, penalty: str):
        Xs = StandardScaler().fit_transform(X)
        with warnings.catch_warnings():  # sklearn>=1.8 deprecates `penalty`; liblinear still honours it
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=FutureWarning)
            selector = SelectFromModel(
                LogisticRegression(penalty=penalty, solver="liblinear", random_state=self.random_state),
                threshold="0.1*mean",
            ).fit(Xs, y)
        return pd.Series(selector.get_support(), index=X.columns)

    def _l1(self, X, y, Xv, yv):
        return self._regularization(X, y, "l1")

    def _l2(self, X, y, Xv, yv):
        return self._regularization(X, y, "l2")

    def _rfe(self, X, y, Xv, yv):
        rfe = RFE(
            DecisionTreeClassifier(max_depth=6, random_state=self.random_state),
            n_features_to_select=min(self.top_k, X.shape[1]),
        ).fit(X, y)
        return pd.Series(rfe.ranking_ == 1, index=X.columns)

    # ---------------------------------------------------------------- API
    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val) -> SoftFeatureSelector:
        self._encoder = _Encoder(X_train, self.cat_cols)
        X, Xv = self._encoder(X_train), self._encoder(X_val)
        y, yv = np.asarray(y_train), np.asarray(y_val)
        if hasattr(self, "_forest_importance"):
            del self._forest_importance

        votes = pd.DataFrame(index=X.columns)
        for name in self.criteria:
            votes[name] = getattr(self, f"_{name}")(X, y, Xv, yv).reindex(X.columns).fillna(False).astype(bool)
        votes["n_votes"] = votes[list(self.criteria)].sum(axis=1).astype(int)

        self.votes_ = votes.sort_index().sort_values("n_votes", ascending=False, kind="stable")
        self.selected_ = self.votes_.index[self.votes_["n_votes"] >= self.min_votes].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_].copy()


def forward_selection_gini(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    ordered_features: Sequence[str],
    cat_cols: Sequence[str] = (),
    step: int = 1,
    max_features: int | None = None,
    catboost_params: dict | None = None,
) -> pd.DataFrame:
    """Train CatBoost on the first 1, 1+step, ... features (in the given order) and record val Gini.

    Returns ``DataFrame[n_features, last_feature, gini_val]``.
    """
    from catboost import CatBoostClassifier

    params = {"iterations": 300, "depth": 4, "learning_rate": 0.05, "random_seed": 42, "verbose": False}
    params.update(catboost_params or {})
    feats = list(ordered_features)[: max_features or None]
    sizes = list(range(step, len(feats) + 1, step))
    if feats and (not sizes or sizes[-1] != len(feats)):
        sizes.append(len(feats))

    rows = []
    for k in sizes:
        subset = feats[:k]
        cats = [c for c in subset if c in cat_cols]
        model = CatBoostClassifier(**params, cat_features=cats)
        Xt = X_train[subset].copy()
        Xv = X_val[subset].copy()
        for c in cats:
            Xt[c], Xv[c] = to_category_str(Xt[c]), to_category_str(Xv[c])
        model.fit(Xt, y_train, eval_set=(Xv, y_val))
        gini = 2 * roc_auc_score(y_val, model.predict_proba(Xv)[:, 1]) - 1
        rows.append({"n_features": k, "last_feature": subset[-1], "gini_val": gini})
    return pd.DataFrame(rows, columns=["n_features", "last_feature", "gini_val"])


def best_k(result: pd.DataFrame, tolerance: float = 0.005) -> int:
    """Smallest number of features whose val Gini is within ``tolerance`` of the best Gini."""
    best = result["gini_val"].max()
    return int(result.loc[result["gini_val"] >= best - tolerance, "n_features"].min())
