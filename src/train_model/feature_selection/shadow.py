"""Shadow-feature pruning: keep only features whose importance beats a noise floor.

Idea from AIDflow's ``FeaturesShadowPruner`` (re-implemented for CatBoost). For each of
``n_shadows`` shadow samples, every feature gets a *shadow* copy whose values are randomly permuted
(same distribution, no relation with the label). A CatBoost model is fitted on originals + shadows
in each of ``cv`` folds and the mean |SHAP| of every column is measured on the held-out fold.

The noise floor of one run is the ``quantile`` (default 0.9) of the shadow importances. A feature is
kept when its mean importance over all runs exceeds the mean noise floor.

Compared with the ``tree_vs_random`` criterion of :class:`SoftFeatureSelector` (one random column,
one fit), the floor here comes from as many shadows as features, over several permutations and folds,
so a feature does not pass by luck of a single random draw.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold

from train_model.modeling.train import prepare_features

SHADOW_PREFIX = "shadow__"
_DEFAULT_PARAMS = {"iterations": 200, "depth": 4, "learning_rate": 0.1}


class ShadowFeatureSelector(BaseEstimator, TransformerMixin):
    """Fit with ``fit(X, y)``; read ``report_`` and ``selected_features_``.

    ``report_`` columns: ``feature, mean_importance, std_importance, threshold, share_runs_above, keep``
    where ``share_runs_above`` is the share of (shadow, fold) runs in which the feature beat that
    run's floor — a feature kept with a low share is borderline.
    """

    def __init__(
        self,
        cat_cols: Sequence[str] = (),
        n_shadows: int = 3,
        cv: int = 3,
        quantile: float = 0.9,
        catboost_params: dict | None = None,
        random_state: int = 42,
    ):
        self.cat_cols = cat_cols
        self.n_shadows = n_shadows
        self.cv = cv
        self.quantile = quantile
        self.catboost_params = catboost_params
        self.random_state = random_state

    def _run(self, Xs: pd.DataFrame, y: np.ndarray, cats: list[str], tr, va) -> pd.Series:
        model = CatBoostClassifier(**{**_DEFAULT_PARAMS, **(self.catboost_params or {})},
                                   cat_features=cats, random_seed=self.random_state, verbose=False)
        model.fit(Xs.iloc[tr], y[tr])
        shap = model.get_feature_importance(Pool(Xs.iloc[va], y[va], cat_features=cats), type="ShapValues")
        return pd.Series(np.abs(shap[:, :-1]).mean(axis=0), index=Xs.columns)

    def fit(self, X: pd.DataFrame, y):
        feats = list(X.columns)
        cats = [c for c in feats if c in self.cat_cols or not pd.api.types.is_numeric_dtype(X[c])]
        Xp = prepare_features(X, feats, cats).reset_index(drop=True)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        runs, floors, beats = {}, [], []
        for s in range(max(1, self.n_shadows)):
            shadow = pd.DataFrame({SHADOW_PREFIX + c: Xp[c].to_numpy()[rng.permutation(len(Xp))] for c in feats})
            Xs = pd.concat([Xp, shadow], axis=1)
            all_cats = cats + [SHADOW_PREFIX + c for c in cats]
            folds = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state + s)
            for k, (tr, va) in enumerate(folds.split(Xs, y)):
                imp = self._run(Xs, y, all_cats, tr, va)
                floor = float(np.quantile(imp[shadow.columns], np.clip(self.quantile, 0, 1)))
                runs[f"s{s}_f{k}"] = imp[feats]
                floors.append(floor)
                beats.append(imp[feats] > floor)

        table = pd.DataFrame(runs)
        threshold = float(np.mean(floors))
        report = pd.DataFrame({
            "feature": feats,
            "mean_importance": table.mean(axis=1).to_numpy(),
            "std_importance": table.std(axis=1).to_numpy(),
            "threshold": threshold,
            "share_runs_above": pd.concat(beats, axis=1).mean(axis=1).to_numpy(),
        })
        report["keep"] = report["mean_importance"] > threshold
        self.runs_ = table
        self.threshold_ = threshold
        self.report_ = report.sort_values("mean_importance", ascending=False).reset_index(drop=True)
        self.selected_features_ = [f for f in feats if bool(report.set_index("feature").loc[f, "keep"])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)
