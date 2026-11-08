"""Population Stability Index — the single PSI implementation of the package.

Rule of thumb: PSI < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray, eps: float) -> float:
    e = np.where(expected <= 0, eps, expected)
    a = np.where(actual <= 0, eps, actual)
    return float(np.sum((a - e) * np.log(a / e)))


def psi(
    expected,
    actual,
    bins: int = 10,
    method: Literal["quantile", "uniform"] = "quantile",
    eps: float = 1e-6,
) -> float:
    """PSI between two numeric samples. NaN values are dropped; returns NaN if either side is empty.

    * ``quantile``: bin edges = quantiles of ``expected`` (robust to any scale, e.g. scores 300–850).
    * ``uniform``: equal-width bins over the combined min/max.
    """
    e = pd.to_numeric(pd.Series(np.asarray(expected, dtype=float)), errors="coerce").dropna().to_numpy()
    a = pd.to_numeric(pd.Series(np.asarray(actual, dtype=float)), errors="coerce").dropna().to_numpy()
    if len(e) == 0 or len(a) == 0:
        return float("nan")

    if method == "quantile":
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    elif method == "uniform":
        lo, hi = min(e.min(), a.min()), max(e.max(), a.max())
        edges = np.unique(np.linspace(lo, hi, bins + 1))
    else:
        raise ValueError(f"method must be 'quantile' or 'uniform', got {method!r}")
    if len(edges) < 2:
        return 0.0 if np.allclose(a, e[0]) else _psi_from_shares(np.array([1.0, 0.0]), np.array(
            [np.mean(a == e[0]), np.mean(a != e[0])]), eps)

    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(e, bins=edges)[0] / len(e)
    a_share = np.histogram(a, bins=edges)[0] / len(a)
    return _psi_from_shares(e_share, a_share, eps)


def psi_categorical(expected, actual, eps: float = 1e-6) -> float:
    """PSI over category frequencies (NaN treated as its own category)."""
    e = pd.Series(expected).astype(str).value_counts(normalize=True)
    a = pd.Series(actual).astype(str).value_counts(normalize=True)
    if e.empty or a.empty:
        return float("nan")
    cats = e.index.union(a.index)
    return _psi_from_shares(e.reindex(cats, fill_value=0).to_numpy(), a.reindex(cats, fill_value=0).to_numpy(), eps)


def psi_by_feature(df_ref: pd.DataFrame, df_cur: pd.DataFrame, features: Sequence[str], bins: int = 10) -> pd.DataFrame:
    """PSI of each feature between a reference and a current frame (numeric: quantile PSI; else categorical)."""
    rows = []
    for f in features:
        if pd.api.types.is_numeric_dtype(df_ref[f]):
            value = psi(df_ref[f], df_cur[f], bins=bins)
        else:
            value = psi_categorical(df_ref[f], df_cur[f])
        rows.append({"feature": f, "psi": value})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def psi_by_bucket(ref_scores, cur_scores, bucket_edges: Sequence[float], eps: float = 1e-6) -> pd.DataFrame:
    """Per-bucket PSI contribution for fixed score buckets (e.g. ``[300, 400, ..., 850]``)."""
    edges = np.asarray(bucket_edges, dtype=float)
    labels = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
    ref = pd.cut(np.asarray(ref_scores, dtype=float), edges, labels=labels, right=False, include_lowest=True)
    cur = pd.cut(np.asarray(cur_scores, dtype=float), edges, labels=labels, right=False, include_lowest=True)
    r = pd.Series(ref).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    c = pd.Series(cur).value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy()
    rr, cc = np.where(r <= 0, eps, r), np.where(c <= 0, eps, c)
    return pd.DataFrame({"bucket": labels, "ref_pct": r, "cur_pct": c, "psi": (cc - rr) * np.log(cc / rr)})
