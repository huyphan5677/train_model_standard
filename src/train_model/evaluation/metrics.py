"""Discrimination metrics: AUC / Gini (overall and per group), decile (lift) table, PR-AUC per bin."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


def auc_gini(y_true, y_score) -> dict[str, float]:
    """``{"auc": AUC, "gini": 2*AUC - 1}``."""
    a = float(roc_auc_score(y_true, y_score))
    return {"auc": a, "gini": 2 * a - 1}


def auc_gini_by_group(df: pd.DataFrame, y_true_col: str, y_score_col: str, group_col: str) -> pd.DataFrame:
    """AUC / Gini per group (e.g. client or month). Groups with a single class get NaN."""
    rows = []
    for group, g in df.groupby(group_col, sort=True):
        if g[y_true_col].nunique() < 2:
            rows.append({"group": group, "n": len(g), "auc": np.nan, "gini": np.nan})
        else:
            rows.append({"group": group, "n": len(g), **auc_gini(g[y_true_col], g[y_score_col])})
    return pd.DataFrame(rows, columns=["group", "n", "auc", "gini"])


def decile_table(y_true, y_score, n_bins: int = 10, score=None) -> pd.DataFrame:
    """Lift table: bin ``y_score`` into quantiles, ``top=1`` = highest scores.

    Columns: ``top, count, n_positive, acc_positive, positive_rate, acc_positive_rate,
    cover_positive, proba_range`` (+ ``score_range`` when ``score`` is given).
    Ties in ``y_score`` may produce fewer than ``n_bins`` rows.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score, dtype=float)})
    if score is not None:
        df["s"] = np.asarray(score, dtype=float)
    codes = pd.qcut(df["p"], q=n_bins, labels=False, duplicates="drop")
    df["top"] = codes.max() - codes + 1

    agg = {"count": ("y", "size"), "n_positive": ("y", "sum"), "p_min": ("p", "min"), "p_max": ("p", "max")}
    if score is not None:
        agg.update(s_min=("s", "min"), s_max=("s", "max"))
    t = df.groupby("top").agg(**agg).reset_index().sort_values("top")

    t["acc_positive"] = t["n_positive"].cumsum()
    t["positive_rate"] = t["n_positive"] / t["count"]
    t["acc_positive_rate"] = t["acc_positive"] / t["count"].cumsum()
    total = t["n_positive"].sum()
    t["cover_positive"] = t["acc_positive"] / total if total else np.nan
    t["proba_range"] = [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t["p_min"], t["p_max"])]
    cols = ["top", "count", "n_positive", "acc_positive", "positive_rate", "acc_positive_rate",
            "cover_positive", "proba_range"]
    if score is not None:
        t["score_range"] = [f"[{a:.0f}, {b:.0f}]" for a, b in zip(t["s_min"], t["s_max"])]
        cols.append("score_range")
    return t[cols].reset_index(drop=True)


def pr_auc_by_bin(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Precision-recall AUC inside each quantile bin of ``y_score`` (NaN for single-class bins)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score, dtype=float)})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    rows = []
    for b, g in df.groupby("bin", observed=True):
        if g["y"].nunique() < 2:
            value = np.nan
        else:
            precision, recall, _ = precision_recall_curve(g["y"], g["p"])
            value = float(auc(recall, precision))
        rows.append({"bin": str(b), "pr_auc": value})
    return pd.DataFrame(rows, columns=["bin", "pr_auc"])
