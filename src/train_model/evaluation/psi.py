"""Population Stability Index — volume PSI, event (lift) PSI and an importance-weighted drift report.

Rule of thumb: PSI < 0.1 stable, 0.1–0.25 moderate shift, > 0.25 significant shift.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from train_model.preprocessing.fill import to_category_str


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
    e = to_category_str(pd.Series(expected)).value_counts(normalize=True)
    a = to_category_str(pd.Series(actual)).value_counts(normalize=True)
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


# --------------------------------------------------------------------------- event (lift) PSI
def _bin_labels(ref: pd.Series, cur: pd.Series, bins: int) -> tuple[pd.Series, pd.Series]:
    """Shared bins: quantile edges of ``ref`` for numerics, categories (NaN = own category) otherwise."""
    if pd.api.types.is_numeric_dtype(ref):
        r = pd.to_numeric(ref, errors="coerce")
        edges = np.unique(np.quantile(r.dropna(), np.linspace(0, 1, bins + 1))) if r.notna().any() else np.array([0.0])
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        edges[0], edges[-1] = -np.inf, np.inf

        def cut(s: pd.Series) -> pd.Series:
            codes = pd.cut(pd.to_numeric(s, errors="coerce"), edges, labels=False, include_lowest=True)
            return codes.map(lambda c: "missing" if pd.isna(c) else f"b{int(c):02d}")

        return cut(ref), cut(cur)
    return to_category_str(ref), to_category_str(cur)


def event_psi(ref_values, ref_y, cur_values, cur_y, bins: int = 10, min_count: int = 30, eps: float = 1e-6,
              event_label: int = 1) -> float:
    """PSI of the distribution of the **event rows** (``y == event_label``) across bins ("lift PSI" in AIDflow).

    Volume PSI can stay low while the risk moves: e.g. the bads shift from one bin to another.
    This compares, per bin, the share of all event rows that fall into the bin. Bins with
    ``count <= min_count`` on either side are ignored (unstable rates). When label 1 means *good*
    (``target_label: good``), pass ``event_label=0`` to follow the bads.
    """
    r_bin, c_bin = _bin_labels(pd.Series(ref_values).reset_index(drop=True),
                               pd.Series(cur_values).reset_index(drop=True), bins)
    ry = pd.Series((np.asarray(ref_y) == event_label).astype(float))
    cy = pd.Series((np.asarray(cur_y) == event_label).astype(float))
    r = ry.groupby(r_bin.to_numpy()).agg(["sum", "count"])
    c = cy.groupby(c_bin.to_numpy()).agg(["sum", "count"])
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0), c.reindex(idx, fill_value=0)
    ok = (r["count"] > min_count) & (c["count"] > min_count)
    if not ok.any() or r.loc[ok, "sum"].sum() == 0 or c.loc[ok, "sum"].sum() == 0:
        return float("nan")
    e = (r.loc[ok, "sum"] / r.loc[ok, "sum"].sum()).to_numpy()
    a = (c.loc[ok, "sum"] / c.loc[ok, "sum"].sum()).to_numpy()
    return _psi_from_shares(e, a, eps)


def _status(v: float) -> str:
    if np.isnan(v):
        return "n/a"
    return "stable" if v < 0.1 else ("warning" if v < 0.25 else "critical")


def psi_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    features: Sequence[str],
    target: str,
    importance: pd.Series | dict | None = None,
    bins: int = 10,
    min_count: int = 30,
    event_label: int = 1,
) -> pd.DataFrame:
    """Per-feature drift: volume PSI + event PSI (rows with ``target == event_label``), optionally
    weighted by model importance.

    ``importance`` (e.g. mean |SHAP| or |coef|) is normalised to sum 1 over ``features``; the weighted
    columns show how much each feature's drift matters *for the model*, and their sums (last row
    ``(model)``) give a model-level drift figure. Status: stable < 0.1 <= warning < 0.25 <= critical.
    """
    rows = []
    for f in features:
        r_bin, c_bin = _bin_labels(df_ref[f].reset_index(drop=True), df_cur[f].reset_index(drop=True), bins)
        e = r_bin.value_counts(normalize=True)
        a = c_bin.value_counts(normalize=True)
        idx = e.index.union(a.index)
        vol = _psi_from_shares(e.reindex(idx, fill_value=0).to_numpy(), a.reindex(idx, fill_value=0).to_numpy(), 1e-6)
        ev = event_psi(df_ref[f], df_ref[target], df_cur[f], df_cur[target], bins=bins, min_count=min_count,
                       event_label=event_label)
        rows.append({"feature": f, "psi": vol, "event_psi": ev})
    out = pd.DataFrame(rows)
    out["psi_status"] = out["psi"].map(_status)
    out["event_psi_status"] = out["event_psi"].map(_status)
    if importance is not None:
        w = pd.Series(importance, dtype=float).reindex(features).abs().fillna(0.0)
        w = w / w.sum() if w.sum() > 0 else w
        out["weight"] = w.to_numpy()
        out["weighted_psi"] = out["psi"] * out["weight"]
        out["weighted_event_psi"] = out["event_psi"].fillna(0.0) * out["weight"]
        out = out.sort_values("weight", ascending=False).reset_index(drop=True)
        total = {"feature": "(model)", "weight": out["weight"].sum(),
                 "weighted_psi": out["weighted_psi"].sum(), "weighted_event_psi": out["weighted_event_psi"].sum()}
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    return out
