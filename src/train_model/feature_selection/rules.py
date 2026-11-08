"""Interaction rules mined from shallow boosted trees (RuleFit-style), for transparent models.

Idea from AIDflow's ``TreeMiner`` (re-implemented on scikit-learn trees). A tree ensemble captures
interactions a linear scorecard cannot; this module turns them into readable binary features
("rule holds = 1") that can be added to a logistic scorecard.

1. **Extract** — fit a shallow ``GradientBoostingClassifier``; every root-to-leaf path is a rule
   (conjunction of ``feature <= t`` / ``feature > t``). Redundant bounds on the same feature are merged.
   Only rules with at least ``min_rule_length`` distinct features are kept (default 2 = interactions;
   single-feature rules duplicate what binning already does). Duplicates are removed.
2. **Support** — drop rules matching < ``min_rule_support`` of rows or containing fewer label-1 rows
   than ``min_target_support`` × (all label-1 rows), clipped to [1 %, 50 %] of them.
3. **Simpson overlap** — ``|A∩B| / min(|A|, |B|) > overlap_threshold`` means one rule is (almost) inside
   the other: keep the larger / shorter one, unless the two label rates differ significantly
   (two-proportion z-test at ``overlap_alpha``) — then both carry information.
4. **Phi correlation** — drop a rule whose correlation with an earlier rule exceeds ``max_correlation``.
5. **Lasso** — ``LassoCV`` on rule columns scaled by ``length^-length_penalty`` (longer rules are
   penalised more, Friedman & Popescu 2008); the penalty is chosen by the *1-SE rule* (largest alpha
   whose CV error is within one standard error of the minimum, as glmnet's ``lambda.1se``) rather
   than the CV minimum, which tends to let a few noise rules in; keep up to ``max_rules`` rules with
   non-zero coefficient.

``base_estimator`` (e.g. the scorecard's own ``LogisticRegression``): the trees start from its
predictions (``init``) and the Lasso explains its residual ``y - p_base``. Rules are then only kept
when they add something the main-effect model misses. Without it (AIDflow's behaviour) the trees
mostly re-discover main effects, and on data with no real interaction the selected rules are noise
that the scorecard already covers.

Input must be numeric without missing values — typically the WoE matrix of a scorecard
(see ``LogisticScorecard(interactions=True)``, which also translates WoE thresholds back to bins).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import Lasso, LassoCV


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str  # "<=" or ">"
    threshold: float

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.feature].to_numpy(dtype=float)
        return x <= self.threshold if self.op == "<=" else x > self.threshold

    def __str__(self) -> str:
        return f"{self.feature} {self.op} {self.threshold:.4g}"


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        mask = np.ones(len(X), dtype=bool)
        for c in self.conditions:
            mask &= c.apply(X)
        return mask

    @property
    def features(self) -> list[str]:
        return sorted({c.feature for c in self.conditions})

    def __len__(self) -> int:
        return len(self.features)

    def __str__(self) -> str:
        return " & ".join(map(str, self.conditions))


def simplify(conditions) -> Rule:
    """Merge bounds per feature (tightest ``<=`` and ``>``) and sort, so equal rules compare equal."""
    upper: dict[str, float] = {}
    lower: dict[str, float] = {}
    for c in conditions:
        if c.op == "<=":
            upper[c.feature] = min(upper.get(c.feature, np.inf), c.threshold)
        else:
            lower[c.feature] = max(lower.get(c.feature, -np.inf), c.threshold)
    out = [Condition(f, ">", t) for f, t in lower.items()] + [Condition(f, "<=", t) for f, t in upper.items()]
    return Rule(tuple(sorted(out, key=lambda c: (c.feature, c.op))))


def extract_rules(trees, feature_names: list[str]) -> list[Rule]:
    """All root-to-leaf paths of fitted scikit-learn decision trees, simplified and de-duplicated."""
    rules: dict[Rule, None] = {}
    for est in trees:
        t = est.tree_

        def walk(node: int, conds: tuple, t=t) -> None:
            left, right = t.children_left[node], t.children_right[node]
            if left == right:  # leaf
                if conds:
                    rules[simplify(conds)] = None
                return
            f, th = feature_names[t.feature[node]], float(t.threshold[node])
            walk(left, conds + (Condition(f, "<=", th),))
            walk(right, conds + (Condition(f, ">", th),))

        walk(0, ())
    return list(rules)


class TreeRuleMiner(BaseEstimator, TransformerMixin):
    """See module docstring. After ``fit``: ``rules_`` (name -> :class:`Rule`), ``stats_`` (table),
    ``n_rules_`` (rule count after each step); ``transform(X)`` returns 0/1 columns ``ixn_00, ixn_01 ...``."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        min_rule_length: int = 2,
        min_rule_support: float = 0.01,
        min_target_support: float = 0.15,
        overlap_threshold: float = 0.9,
        overlap_alpha: float = 0.05,
        max_correlation: float = 0.9,
        length_penalty: float = 0.5,
        max_rules: int = 30,
        cv: int = 3,
        base_estimator=None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_rule_length = min_rule_length
        self.min_rule_support = min_rule_support
        self.min_target_support = min_target_support
        self.overlap_threshold = overlap_threshold
        self.overlap_alpha = overlap_alpha
        self.max_correlation = max_correlation
        self.length_penalty = length_penalty
        self.max_rules = max_rules
        self.cv = cv
        self.base_estimator = base_estimator
        self.random_state = random_state

    # ------------------------------------------------------------------ pruning steps
    def _support(self, M: np.ndarray, y: np.ndarray, idx: list[int]) -> list[int]:
        n_pos = y.sum()
        min_target = np.clip(self.min_target_support * n_pos, 0.01 * n_pos, 0.5 * n_pos)
        share = M[:, idx].mean(axis=0)
        pos = y @ M[:, idx]
        return [i for i, s, p in zip(idx, share, pos) if s > self.min_rule_support and p > min_target]

    def _simpson(self, M: np.ndarray, y: np.ndarray, idx: list[int], rules: list[Rule]) -> list[int]:
        if not idx:
            return idx
        S = M[:, idx].astype(float)
        inter = S.T @ S
        sup = np.diag(inter).copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = inter / np.minimum.outer(sup, sup)
        np.fill_diagonal(sim, 0)
        ypos = y @ S
        z_crit = norm.ppf(1 - self.overlap_alpha / 2)

        def differs(a: int, b: int) -> bool:
            p_pool = (ypos[a] + ypos[b]) / (sup[a] + sup[b])
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / sup[a] + 1 / sup[b]))
            return se > 0 and abs(ypos[a] / sup[a] - ypos[b] / sup[b]) / se > z_crit

        order = sorted(range(len(idx)), key=lambda k: (-sup[k], len(rules[idx[k]])))
        kept, dropped = [], set()
        for a in order:
            if a in dropped:
                continue
            kept.append(a)
            for b in np.where(sim[a] > self.overlap_threshold)[0]:
                if b not in dropped and b != a and not differs(a, b):
                    dropped.add(int(b))
        return [idx[k] for k in sorted(kept)]

    def _phi(self, M: np.ndarray, idx: list[int]) -> list[int]:
        if len(idx) < 2:
            return idx
        corr = np.nan_to_num(np.corrcoef(M[:, idx].T.astype(float)))
        upper = np.triu(corr, k=1)
        worst = np.abs(upper).max(axis=0)  # max |corr| with an earlier rule
        return [i for i, w in zip(idx, worst) if w < self.max_correlation]

    def _lasso(self, M: np.ndarray, target: np.ndarray, idx: list[int], rules: list[Rule]) -> tuple[list[int], np.ndarray]:
        if not idx:
            return idx, np.array([])
        w = np.array([len(rules[i]) for i in idx], dtype=float) ** -self.length_penalty
        Z = M[:, idx] * w
        cv = LassoCV(cv=self.cv, random_state=self.random_state).fit(Z, target)
        # 1-SE rule: the largest alpha whose CV error is within one standard error of the best one
        mse = cv.mse_path_.mean(axis=1)
        se = cv.mse_path_.std(axis=1) / np.sqrt(cv.mse_path_.shape[1])
        best = int(np.argmin(mse))
        alpha = float(cv.alphas_[mse <= mse[best] + se[best]].max())
        coef_all = Lasso(alpha=alpha).fit(Z, target).coef_
        coef = np.abs(coef_all)
        order = [k for k in np.argsort(-coef)[: self.max_rules] if coef[k] > 1e-6]
        return [idx[k] for k in order], coef_all[order]

    # ------------------------------------------------------------------ API
    def fit(self, X: pd.DataFrame, y):
        if X.isna().any().any():
            raise ValueError("TreeRuleMiner needs X without missing values (e.g. a WoE matrix)")
        y = np.asarray(y, dtype=float)
        init = None
        target = y
        if self.base_estimator is not None:
            init = clone(self.base_estimator).fit(X, y)
            target = y - init.predict_proba(X)[:, 1]  # residual of the main-effect model
        gbm = GradientBoostingClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                         learning_rate=self.learning_rate, subsample=self.subsample,
                                         init=init, random_state=self.random_state).fit(X, y)
        all_rules = [r for r in extract_rules(gbm.estimators_[:, 0], list(X.columns)) if len(r) >= self.min_rule_length]
        M = np.column_stack([r.apply(X) for r in all_rules]).astype(np.int8) if all_rules else np.zeros((len(X), 0))
        idx = list(range(len(all_rules)))
        counts = {"extracted": len(idx)}
        idx = self._support(M, y, idx)
        counts["support"] = len(idx)
        idx = self._simpson(M, y, idx, all_rules)
        counts["simpson"] = len(idx)
        idx = self._phi(M, idx)
        counts["phi_corr"] = len(idx)
        idx, coef = self._lasso(M, target, idx, all_rules)
        counts["lasso"] = len(idx)

        self.generator_ = gbm
        self.n_rules_ = counts
        self.rules_ = {f"ixn_{k:02d}": all_rules[i] for k, i in enumerate(idx)}
        Ms = M[:, idx].astype(float)
        self.stats_ = pd.DataFrame({
            "name": list(self.rules_),
            "rule": [str(r) for r in self.rules_.values()],
            "length": [len(r) for r in self.rules_.values()],
            "support": Ms.mean(axis=0) if idx else [],
            "target_rate": (y @ Ms) / Ms.sum(axis=0) if idx else [],
            "n_target": (y @ Ms).astype(int) if idx else [],
            "lasso_coef": coef,
        })
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({n: r.apply(X).astype(np.int8) for n, r in self.rules_.items()}, index=X.index)
