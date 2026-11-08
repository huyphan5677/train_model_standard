"""Cross-audit: can model B find significant risk differences *inside* the bins of model A?

Idea adapted from AIDflow's *ModelsCrossTransition* / *ModelCrossDivergence* (re-implemented).

Cut each model's score into ``n_bins`` quantile bins (bin 0 = lowest score). For an *audited* model A
and an *auditor* model B, cross-tabulate the bins. Inside bin ``i`` of A, model A says every row has
the same event rate ``r_i``. If B's sub-segment ``(i, j)`` has a rate that is significantly different
**and in the direction B predicts** (higher when ``j > i``, lower when ``j < i``), B has found
information that A misses — a blind spot of A.

Divergence of A audited by B (weights = share of all rows in the cell)::

    up   = sum over significant cells with rate > r_i of  w_ij * (rate_ij - r_i)
    down = sum over significant cells with rate < r_i of  w_ij * (rate_ij - r_i)
    total = up - down          (>= 0; 0 = B adds nothing A does not already know)

All scores must have the same orientation (higher = higher probability of label 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class CrossAuditResult:
    pairs: pd.DataFrame  # audited, auditor, up, down, total
    summary: pd.DataFrame  # per model: audited (blind spots found by others, lower = better), auditor (higher = better)


def score_bins(score, n_bins: int = 10) -> np.ndarray:
    """Quantile bin index 0..n_bins-1 (0 = lowest score); ties broken by order."""
    return pd.qcut(pd.Series(np.asarray(score, dtype=float)).rank(method="first"), n_bins, labels=False).to_numpy()


def audit_pair(
    y_true, bins_audited: np.ndarray, bins_auditor: np.ndarray, alpha: float = 0.05, min_cell: int = 30
) -> dict[str, float]:
    """Divergence of one audited model given one auditor (see module docstring)."""
    df = pd.DataFrame({"y": np.asarray(y_true, dtype=float), "a": bins_audited, "b": bins_auditor})
    n = len(df)
    base = df.groupby("a")["y"].mean()
    cell = df.groupby(["a", "b"])["y"].agg(rate="mean", count="size").reset_index()
    cell["base"] = cell["a"].map(base)
    cell = cell[cell["count"] >= min_cell]
    se = np.sqrt((cell["base"] * (1 - cell["base"])).clip(lower=1e-9) / cell["count"])
    d = cell["rate"] - cell["base"]
    sig = (d.abs() / se) > norm.ppf(1 - alpha / 2)
    true_signal = ((d > 0) & (cell["b"] > cell["a"])) | ((d < 0) & (cell["b"] < cell["a"]))
    contrib = (d * cell["count"] / n).where(sig & true_signal, 0.0)
    up = float(contrib[contrib > 0].sum())
    down = float(contrib[contrib < 0].sum())
    return {"up": up, "down": down, "total": up - down}


def cross_audit(
    y_true, scores: Mapping[str, np.ndarray], n_bins: int = 10, alpha: float = 0.05, min_cell: int = 30
) -> CrossAuditResult:
    """Audit every ordered pair of models. ``scores`` maps model name -> score/probability array."""
    if len(scores) < 2:
        raise ValueError("cross_audit needs at least two models")
    bins = {name: score_bins(s, n_bins) for name, s in scores.items()}
    rows = [
        {"audited": a, "auditor": b, **audit_pair(y_true, bins[a], bins[b], alpha, min_cell)}
        for a, b in permutations(scores, 2)
    ]
    pairs = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "audited": pairs.groupby("audited")["total"].mean(),
            "auditor": pairs.groupby("auditor")["total"].mean(),
        }
    ).reindex(list(scores))
    summary.index.name = "model"
    return CrossAuditResult(pairs=pairs, summary=summary)
