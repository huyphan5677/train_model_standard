"""Feature ablation: how much Gini does the model lose without each feature?

Idea from AIDflow's ``FeatureAblation`` (re-implemented).

* :func:`logistic_ablation` — for a logistic regression the refit without feature ``j`` is
  approximated analytically, no refitting loop: the inverse Hessian of the reduced model comes from
  the full one by the deletion (Schur complement) formula, then a couple of Newton steps converge to
  the reduced optimum. Unlike AIDflow (plain MLE), the L2 penalty of scikit-learn's
  ``LogisticRegression`` (``C``) is included in the Hessian and the gradient, so the result matches a
  real refit of the same penalised model.
* :func:`refit_ablation` — any model (e.g. CatBoost): refit once per feature through a callback.

The Gini drop on the evaluation sample is tested with DeLong (full vs. ablated predictions are
correlated), so ``significant`` marks features whose removal hurts beyond noise.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from train_model.evaluation.metrics import auc_gini
from train_model.evaluation.stat_tests import delong_test


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def _row(feature: str, y_eval, p_full: np.ndarray, p_without: np.ndarray, gini_full: float, alpha: float) -> dict:
    g = auc_gini(y_eval, p_without)["gini"]
    test = delong_test(y_eval, p_without, p_full, alpha=alpha)
    delta = g - gini_full
    return {"feature": feature, "gini_full": gini_full, "gini_without": g, "delta_gini": delta,
            "p_value": test["p_value"], "significant": bool(test["significant"] and delta < 0)}


def _finish(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values("delta_gini").reset_index(drop=True)


def logistic_ablation(
    X_train: pd.DataFrame,
    y_train,
    coef: Sequence[float],
    intercept: float,
    X_eval: pd.DataFrame,
    y_eval,
    C: float | None = 1.0,
    n_steps: int = 2,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Ablation of a fitted logistic regression (``coef`` in ``X_train`` column order).

    ``C`` is scikit-learn's inverse regularisation strength (``None`` = unpenalised). Returns one row per
    feature: ``gini_full, gini_without, delta_gini (<0 = feature helps), p_value, significant``,
    most useful feature first.
    """
    feats = list(X_train.columns)
    Xd = np.column_stack([np.ones(len(X_train)), X_train.to_numpy(dtype=float)])
    Xe = np.column_stack([np.ones(len(X_eval)), X_eval[feats].to_numpy(dtype=float)])
    y = np.asarray(y_train, dtype=float)
    beta = np.concatenate([[intercept], np.asarray(coef, dtype=float)])
    pen = np.zeros(len(beta)) if C is None else np.r_[0.0, np.full(len(feats), 1.0 / C)]

    p = _sigmoid(Xd @ beta)
    H = (Xd.T * (p * (1 - p))) @ Xd + np.diag(pen)
    V = np.linalg.pinv(H)
    p_full = _sigmoid(Xe @ beta)
    gini_full = auc_gini(y_eval, p_full)["gini"]

    rows = []
    for j, f in enumerate(feats, start=1):
        keep = np.ones(len(beta), dtype=bool)
        keep[j] = False
        H_inv = V[keep][:, keep] - np.outer(V[keep, j], V[keep, j]) / V[j, j]
        Xr, b, pr = Xd[:, keep], beta[keep].copy(), pen[keep]
        for _ in range(n_steps):
            grad = Xr.T @ (y - _sigmoid(Xr @ b)) - pr * b
            b = b + H_inv @ grad
        rows.append(_row(f, y_eval, p_full, _sigmoid(Xe[:, keep] @ b), gini_full, alpha))
    return _finish(rows)


def refit_ablation(
    fit_predict: Callable[[list[str]], np.ndarray],
    features: Sequence[str],
    y_eval,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Ablation by refitting: ``fit_predict(features) -> predicted probabilities on the eval sample``.

    Called ``len(features) + 1`` times (full model + one per dropped feature).
    """
    feats = list(features)
    p_full = np.asarray(fit_predict(feats), dtype=float)
    gini_full = auc_gini(y_eval, p_full)["gini"]
    rows = [_row(f, y_eval, p_full, np.asarray(fit_predict([x for x in feats if x != f]), dtype=float),
                 gini_full, alpha) for f in feats]
    return _finish(rows)
