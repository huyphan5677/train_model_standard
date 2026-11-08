"""Classic points scorecard: logistic regression on WoE features + a points table per bin.

A transparent challenger to the CatBoost model (cf. the GLM agent of AIDflow): every feature bin
gets a fixed number of points, the score is the sum, and the table can be read by a risk committee.

Scaling (standard PDO convention)::

    factor = pdo / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds of a GOOD customer)

so ``base_score`` points correspond to good:bad odds of ``base_odds`` and every ``pdo`` points the
odds double. ``target_label`` tells whether label 1 is the good (``"good"``) or bad (``"bad"``) class.

``interactions=True`` adds binary interaction rules mined from shallow trees on the WoE matrix
(:class:`~train_model.feature_selection.rules.TreeRuleMiner`, started from the scorecard's own
logistic regression so that only rules adding to the main effects are kept); each selected rule gets
its own row in the points table ("+x points when the rule holds"), described with the original bins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from train_model.feature_selection.binning import WoETransformer
from train_model.feature_selection.rules import Rule, TreeRuleMiner


def _fmt_bin(b) -> str:
    """optbinning bins: interval strings for numerics, arrays of categories for categoricals."""
    if isinstance(b, str):
        return b
    try:
        return "[" + ", ".join(map(str, list(b))) + "]"
    except TypeError:
        return str(b)


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
        interactions: bool = False,
        interaction_params: dict | None = None,
    ):
        self.features = features
        self.cat_cols = cat_cols
        self.max_n_bins = max_n_bins
        self.C = C
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.target_label = target_label
        self.interactions = interactions
        self.interaction_params = interaction_params

    # ------------------------------------------------------------------ fit / predict
    def _woe(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.woe_.transform(X[self.features_]).astype(float).fillna(0.0)

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        """Model matrix: WoE columns (+ interaction rule columns)."""
        W = self._woe(X)
        if self.miner_ is None or not self.miner_.rules_:
            return W
        return pd.concat([W, self.miner_.transform(W).astype(float)], axis=1)

    def fit(self, X: pd.DataFrame, y):
        if self.target_label not in ("good", "bad"):
            raise ValueError("target_label must be 'good' or 'bad'")
        feats = list(self.features) if self.features is not None else list(X.columns)
        cats = [c for c in self.cat_cols if c in feats]
        self.woe_ = WoETransformer(features=feats, cat_cols=cats, max_n_bins=self.max_n_bins).fit(X[feats], y)
        self.features_ = [f for f in feats if f in self.woe_.binnings_]  # binning may fail for some
        self.miner_ = None
        if self.interactions:
            params = {"base_estimator": LogisticRegression(C=self.C, max_iter=2000), **(self.interaction_params or {})}
            self.miner_ = TreeRuleMiner(**params).fit(self._woe(X), np.asarray(y))
        D = self._design(X)
        self.design_columns_ = list(D.columns)
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000).fit(D, np.asarray(y))
        self.classes_ = self.lr_.classes_
        self.coef_ = pd.Series(self.lr_.coef_[0], index=self.design_columns_, name="coef")
        self.intercept_ = float(self.lr_.intercept_[0])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self._design(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict(self._design(X))

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
        logit = self.lr_.decision_function(self._design(X))
        return self._offset + self._factor * self._sign * logit

    def _bin_table(self, feature: str) -> pd.DataFrame:
        table = self.woe_.binnings_[feature].binning_table.build()
        table = table[table.index != "Totals"].copy()
        table["woe_num"] = pd.to_numeric(table["WoE"], errors="coerce")
        return table[table["woe_num"].notna()]

    def describe_rule(self, rule: Rule) -> str:
        """Translate WoE thresholds of a rule into the original bins, e.g. ``sub_type in {['postpaid']}``."""
        parts = []
        for f in rule.features:
            t = self._bin_table(f)
            t = t[t["Count"] > 0]
            ok = np.ones(len(t), dtype=bool)
            for c in rule.conditions:
                if c.feature == f:
                    ok &= (t["woe_num"] <= c.threshold).to_numpy() if c.op == "<=" else (t["woe_num"] > c.threshold).to_numpy()
            bins = [_fmt_bin(b) for b in t.loc[ok, "Bin"]]
            parts.append(f"{f} in {{{', '.join(bins)}}}")
        return " AND ".join(parts)

    def points_table(self) -> pd.DataFrame:
        """One row per (feature, bin): ``woe, coef, points``, a ``(base)`` row with the intercept part and,
        with interactions, one row per rule (``points`` are added when the rule holds).

        Total points of a customer = base points + the points of the bin they fall into for each feature
        + the points of every rule that holds.
        """
        rows = [{"feature": "(base)", "bin": "", "count": np.nan, "event_rate": np.nan, "woe": np.nan,
                 "coef": self.intercept_,
                 "points": self._offset + self._factor * self._sign * self.intercept_}]
        for f in self.features_:
            coef = float(self.coef_[f])
            for _, r in self._bin_table(f).iterrows():
                rows.append({
                    "feature": f,
                    "bin": _fmt_bin(r["Bin"]),
                    "count": r["Count"],
                    "event_rate": r["Event rate"],
                    "woe": float(r["woe_num"]),
                    "coef": coef,
                    "points": self._factor * self._sign * coef * float(r["woe_num"]),
                })
        if self.miner_ is not None:
            stats = self.miner_.stats_.set_index("name")
            for name, rule in self.miner_.rules_.items():
                coef = float(self.coef_[name])
                rows.append({
                    "feature": name,
                    "bin": self.describe_rule(rule),
                    "count": np.nan,
                    "event_rate": stats.loc[name, "target_rate"],
                    "woe": np.nan,
                    "coef": coef,
                    "points": self._factor * self._sign * coef,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ diagnostics
    def ablation(self, X_train: pd.DataFrame, y_train, X_eval: pd.DataFrame, y_eval, alpha: float = 0.05) -> pd.DataFrame:
        """Gini lost on ``X_eval`` without each WoE feature / rule (analytic refit, see
        :func:`~train_model.evaluation.ablation.logistic_ablation`)."""
        from train_model.evaluation.ablation import logistic_ablation

        return logistic_ablation(self._design(X_train), y_train, self.coef_.to_numpy(), self.intercept_,
                                 self._design(X_eval), y_eval, C=self.C, alpha=alpha)
