"""Champion / challenger comparison on several orthogonal criteria.

Idea adapted from AIDflow's *AgentsComparator* (re-implemented). Instead of picking the model with
the best Gini alone, each model is measured on:

==================  ==============================================================  ==========
metric              meaning                                                         better
==================  ==============================================================  ==========
``gini``            discrimination                                                  higher
``ks``              max separation of the cumulative distributions                  higher
``calibration``     share of probability bins with no significant gap               higher
``audited``         blind spots other models find inside this model's bins          lower
``auditor``         blind spots this model finds in the other models                higher
``exposure``        effective number of features = exp(entropy of importance)      lower
``max_share``       share of the single most important feature                      lower
==================  ==============================================================  ==========

Each metric is scaled *leader-anchored*: the best model gets 1.0, the others a proportion of it
(``v / best`` for higher-is-better, ``best / v`` for lower-is-better). The composite is the weighted
mean of the scaled metrics; the highest composite is the champion. Metrics that are missing for any
model (e.g. no importances given) are left out of the composite.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from train_model.evaluation.calibration import calibration_summary
from train_model.evaluation.cross_audit import CrossAuditResult, cross_audit
from train_model.evaluation.metrics import auc_gini, ks_statistic

HIGHER_BETTER = ("gini", "ks", "calibration", "auditor")
LOWER_BETTER = ("audited", "exposure", "max_share")


@dataclass
class ComparisonResult:
    metrics: pd.DataFrame  # raw metric values, one row per model
    scaled: pd.DataFrame  # leader-anchored 0..1 values + composite, sorted best first
    champion: str
    audit: CrossAuditResult | None


def feature_exposure(importance: Mapping[str, float] | pd.Series) -> tuple[float, float]:
    """``(effective number of features, share of the top feature)`` from any importance vector."""
    v = np.abs(np.asarray(list(importance.values()) if isinstance(importance, Mapping) else importance, dtype=float))
    v = v[v > 0]
    if len(v) == 0:
        return float("nan"), float("nan")
    s = v / v.sum()
    return float(np.exp(-(s * np.log(s)).sum())), float(s.max())


def _scale(col: pd.Series, higher_better: bool) -> pd.Series:
    """Leader-anchored scaling to (0, 1], best model = 1.

    Higher-better: ``v / max``. Lower-better: ``min / v``; when the best value is 0 (e.g. no blind
    spot at all) the ratio is undefined, so fall back to ``1 - v / max``.
    """
    col = col.astype(float)
    if higher_better:
        best = col.max()
        return col / best if best > 0 else pd.Series(1.0, index=col.index)
    best, worst = col.min(), col.max()
    if worst <= 0:
        return pd.Series(1.0, index=col.index)
    if best > 0:
        return best / col
    return 1 - col / worst


def compare_models(
    y_true,
    scores: Mapping[str, np.ndarray],
    importances: Mapping[str, Mapping[str, float] | pd.Series] | None = None,
    weights: Mapping[str, float] | None = None,
    n_bins: int = 10,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Compare models scored on the **same rows** (typically the out-of-time test set).

    ``scores``: model name -> predicted probability of label 1 (all with the same orientation).
    ``importances``: optional model name -> feature importance (enables ``exposure`` / ``max_share``).
    ``weights``: optional metric -> weight (default: equal weights).
    """
    y = np.asarray(y_true)
    rows = {}
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        rows[name] = {
            "gini": auc_gini(y, s)["gini"],
            "ks": ks_statistic(y, s),
            "calibration": calibration_summary(y, s, n_bins=n_bins, alpha=alpha)["calibration_score"],
        }
        if importances and name in importances:
            rows[name]["exposure"], rows[name]["max_share"] = feature_exposure(importances[name])

    audit = None
    if len(scores) > 1:
        audit = cross_audit(y, scores, n_bins=n_bins, alpha=alpha)
        for name in scores:
            rows[name]["audited"] = audit.summary.loc[name, "audited"]
            rows[name]["auditor"] = audit.summary.loc[name, "auditor"]

    metrics = pd.DataFrame.from_dict(rows, orient="index")
    metrics.index.name = "model"
    usable = [c for c in (*HIGHER_BETTER, *LOWER_BETTER) if c in metrics and metrics[c].notna().all()]
    scaled = pd.DataFrame({c: _scale(metrics[c], c in HIGHER_BETTER) for c in usable})
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in usable}, dtype=float)
    scaled["composite"] = (scaled[usable] * w).sum(axis=1) / w.sum()
    scaled = scaled.sort_values("composite", ascending=False)
    return ComparisonResult(metrics=metrics, scaled=scaled, champion=str(scaled.index[0]), audit=audit)
