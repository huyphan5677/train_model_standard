"""Classic points scorecard: logistic regression on WoE features + a points table per bin.

A transparent challenger to the CatBoost model (cf. the GLM agent of AIDflow): every feature bin
gets a fixed number of points, the score is the sum, and the table can be read by a risk committee.

Scaling (standard PDO convention)::

    factor = pdo / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds of a GOOD customer)

so ``base_score`` points correspond to good:bad odds of ``base_odds`` and every ``pdo`` points the
odds double. ``target_label`` tells whether label 1 is the good (``"good"``) or bad (``"bad"``) class.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from train_model.feature_selection.binning import WoETransformer


class LogisticScorecard(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        features: Sequence[str] | None = None,
        cat_cols: Sequence[str] = (),
        max_n_bins: int = 10,
        C: float = 1.0,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        target_label: Literal["good", "bad"] = "good",
    ):
        self.features = features
        self.cat_cols = cat_cols
        self.max_n_bins = max_n_bins
        self.C = C
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.target_label = target_label

    # ------------------------------------------------------------------ fit / predict
    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.woe_.transform(X[self.features_]).astype(float).fillna(0.0)

    def fit(self, X: pd.DataFrame, y):
        if self.target_label not in ("good", "bad"):
            raise ValueError("target_label must be 'good' or 'bad'")
        feats = list(self.features) if self.features is not None else list(X.columns)
        cats = [c for c in self.cat_cols if c in feats]
        self.woe_ = WoETransformer(features=feats, cat_cols=cats, max_n_bins=self.max_n_bins).fit(X[feats], y)
        self.features_ = [f for f in feats if f in self.woe_.binnings_]  # binning may fail for some
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(self._woe(X), np.asarray(y))
        self.classes_ = self.lr_.classes_
        self.coef_ = pd.Series(self.lr_.coef_[0], index=self.features_, name="coef")
        self.intercept_ = float(self.lr_.intercept_[0])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self._woe(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict(self._woe(X))

    # ------------------------------------------------------------------ points
    @property
    def _factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def _offset(self) -> float:
        return self.base_score - self._factor * np.log(self.base_odds)

    @property
    def _sign(self) -> float:
        """+1 when label 1 = good (logit is already log-odds of good), -1 otherwise."""
        return 1.0 if self.target_label == "good" else -1.0

    def points(self, X: pd.DataFrame) -> np.ndarray:
        """Total points per row (higher = better customer)."""
        logit = self.lr_.decision_function(self._woe(X))
        return self._offset + self._factor * self._sign * logit

    def points_table(self) -> pd.DataFrame:
        """One row per (feature, bin): ``woe, coef, points`` plus a ``(base)`` row with the intercept part.

        Total points of a customer = base points + the points of the bin they fall into for each feature.
        """
        rows = [{"feature": "(base)", "bin": "", "count": np.nan, "event_rate": np.nan, "woe": np.nan,
                 "coef": self.intercept_,
                 "points": self._offset + self._factor * self._sign * self.intercept_}]
        for f in self.features_:
            table = self.woe_.binnings_[f].binning_table.build()
            table = table[table.index != "Totals"]
            coef = float(self.coef_[f])
            for _, r in table.iterrows():
                woe = pd.to_numeric(r["WoE"], errors="coerce")
                if pd.isna(woe):
                    continue
                rows.append({
                    "feature": f,
                    "bin": str(r["Bin"]),
                    "count": r["Count"],
                    "event_rate": r["Event rate"],
                    "woe": float(woe),
                    "coef": coef,
                    "points": self._factor * self._sign * coef * float(woe),
                })
        return pd.DataFrame(rows)
