"""DeLong test: is the AUC / Gini difference between two models on the same rows real or noise?

Two models scored on the *same* observations have correlated AUC estimates, so comparing their
confidence intervals independently is wrong. DeLong (1988) estimates the covariance of the two AUCs
from the per-observation "structural components" (placement values) and gives a z-test for
``AUC_1 - AUC_2``. Same method as AIDflow's ``TestModelDiff.delong`` (re-implemented), extended with
confidence intervals and a pairwise table.

Gini = 2 * AUC - 1, so the Gini difference is twice the AUC difference with the same z / p-value.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _placements(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-model AUC and DeLong structural components (V10 over positives, V01 over negatives).

    ``scores`` has shape (k, n). Ties count 1/2 (mid-ranks).
    """
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("y_true must contain both classes 0 and 1")
    k = scores.shape[0]
    v10, v01, aucs = np.zeros((k, m)), np.zeros((k, n)), np.zeros(k)
    for i in range(k):
        r_all = rankdata(scores[i], method="average")
        r_pos = rankdata(scores[i][pos], method="average")
        r_neg = rankdata(scores[i][neg], method="average")
        v10[i] = (r_all[pos] - r_pos) / n  # share of negatives below each positive
        v01[i] = 1 - (r_all[neg] - r_neg) / m  # share of positives above each negative
        aucs[i] = v10[i].mean()
    return aucs, v10, v01


def _cov(v10: np.ndarray, v01: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.cov(v10)) / v10.shape[1] + np.atleast_2d(np.cov(v01)) / v01.shape[1]


def auc_ci(y_true, y_score, alpha: float = 0.05) -> dict[str, float]:
    """AUC / Gini with a DeLong confidence interval (level ``1 - alpha``)."""
    y = np.asarray(y_true).astype(int)
    aucs, v10, v01 = _placements(y, np.asarray(y_score, dtype=float)[None, :])
    se = float(np.sqrt(max(_cov(v10, v01)[0, 0], 0.0)))
    z = norm.ppf(1 - alpha / 2)
    lo, hi = max(aucs[0] - z * se, 0.0), min(aucs[0] + z * se, 1.0)
    return {"auc": float(aucs[0]), "auc_se": se, "auc_low": lo, "auc_high": hi,
            "gini": 2 * float(aucs[0]) - 1, "gini_low": 2 * lo - 1, "gini_high": 2 * hi - 1}


def delong_test(y_true, score_1, score_2, alpha: float = 0.05) -> dict[str, float]:
    """Two-sided DeLong test of ``H0: AUC_1 = AUC_2`` for two models scored on the same rows.

    Returns ``auc_1, auc_2, gini_diff (= gini_1 - gini_2), gini_diff_low/high, z, p_value, significant``.
    """
    y = np.asarray(y_true).astype(int)
    s = np.vstack([np.asarray(score_1, dtype=float), np.asarray(score_2, dtype=float)])
    if s.shape[1] != len(y):
        raise ValueError("scores and y_true must have the same length")
    aucs, v10, v01 = _placements(y, s)
    c = _cov(v10, v01)
    diff = aucs[0] - aucs[1]
    var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    if var <= 0:  # identical rankings
        z, p, se = 0.0, 1.0, 0.0
    else:
        se = float(np.sqrt(var))
        z = float(diff / se)
        p = float(2 * norm.sf(abs(z)))
    zc = norm.ppf(1 - alpha / 2)
    return {
        "auc_1": float(aucs[0]),
        "auc_2": float(aucs[1]),
        "gini_diff": 2 * float(diff),
        "gini_diff_low": 2 * float(diff - zc * se),
        "gini_diff_high": 2 * float(diff + zc * se),
        "z": z,
        "p_value": p,
        "significant": bool(p < alpha),
    }


def pairwise_delong(y_true, scores: Mapping[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """DeLong test for every pair of models. One row per pair (``model_1``, ``model_2``)."""
    rows = [{"model_1": a, "model_2": b, **delong_test(y_true, scores[a], scores[b], alpha)}
            for a, b in combinations(scores, 2)]
    return pd.DataFrame(rows)
