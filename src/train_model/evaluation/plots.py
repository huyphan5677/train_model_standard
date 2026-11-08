"""Plot helpers. Every function returns a matplotlib Figure (it never calls ``plt.show()``)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import roc_auc_score, roc_curve


def _roc_axes(ax, title: str) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")


def plot_roc(datasets: dict[str, tuple], title: str = "ROC curves") -> Figure:
    """``datasets = {"train": (y_true, y_score), "test": (...)}``."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y, p) in datasets.items():
        fpr, tpr, _ = roc_curve(y, p)
        a = roc_auc_score(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}: AUC={a:.3f} | Gini={2 * a - 1:.3f}")
    _roc_axes(ax, title)
    fig.tight_layout()
    return fig


def plot_roc_by_group(df: pd.DataFrame, y_true_col: str, y_score_col: str, group_col: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    for group, g in df.groupby(group_col):
        if g[y_true_col].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g[y_true_col], g[y_score_col])
        a = roc_auc_score(g[y_true_col], g[y_score_col])
        ax.plot(fpr, tpr, label=f"{group_col}={group} (Gini={2 * a - 1:.3f})")
    _roc_axes(ax, f"ROC by {group_col}")
    fig.tight_layout()
    return fig


def plot_shap_summary(model, X: pd.DataFrame, max_display: int = 20) -> Figure:
    """SHAP beeswarm summary plot."""
    import shap

    from train_model.evaluation.shap_utils import _shap_values

    values = _shap_values(model, X)
    shap.summary_plot(values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def plot_forward_gini(result: pd.DataFrame, best: int | None = None) -> Figure:
    """Validation Gini vs number of features (output of ``forward_selection_gini``)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(result["n_features"], result["gini_val"], marker="o")
    if best is not None:
        ax.axvline(best, color="tab:red", linestyle="--", label=f"best_k = {best}")
        ax.legend()
    ax.set_xlabel("number of features (in vote order)")
    ax.set_ylabel("Gini (validation)")
    ax.set_title("Forward selection")
    ax.grid(alpha=0.3)
    if len(result) <= 40:
        ax.set_xticks(np.asarray(result["n_features"]))
    fig.tight_layout()
    return fig


def plot_calibration(datasets: dict[str, tuple], n_bins: int = 10, title: str = "Calibration") -> Figure:
    """Mean predicted probability vs observed rate per quantile bin.

    ``datasets = {"catboost": (y_true, y_proba), "scorecard": (...)}``; markers are hollow where the
    gap is not significant (see :func:`~train_model.evaluation.calibration.calibration_table`).
    """
    from train_model.evaluation.calibration import calibration_table

    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = 1.0, 0.0
    for name, (y, p) in datasets.items():
        t = calibration_table(y, p, n_bins=n_bins)
        (line,) = ax.plot(t["proba_mean"], t["event_rate"], linewidth=2, label=name)
        sig = t["significant"].to_numpy()
        ax.scatter(t["proba_mean"][sig], t["event_rate"][sig], s=40, color=line.get_color(), zorder=3)
        ax.scatter(t["proba_mean"][~sig], t["event_rate"][~sig], s=40, facecolors="none",
                   edgecolors=line.get_color(), zorder=3)
        lo = min(lo, t["proba_mean"].min(), t["event_rate"].min())
        hi = max(hi, t["proba_mean"].max(), t["event_rate"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title(f"{title}  (filled = significant gap)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_temporal_consistency(table: pd.DataFrame, feature: str = "", score: float | None = None) -> Figure:
    """Target ratio ``TR = bin rate / period rate - 1`` of each bin over time.

    ``table`` is ``TemporalStabilitySelector.tables_[feature]`` (or ``ConsistencyResult.table``).
    A stable feature has lines that stay on one side of 0.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if table.empty:
        ax.text(0.5, 0.5, "no significant bin", ha="center", va="center", transform=ax.transAxes)
    else:
        periods = sorted(table["t"].unique())
        pos = {t: i for i, t in enumerate(periods)}
        for b, g in table.groupby("bin"):
            g = g.sort_values("t")
            ax.plot([pos[t] for t in g["t"]], g["tr"], marker="o", markersize=3, linewidth=1.5, label=str(b))
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([str(t) for t in periods], rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=7, ncol=2, loc="best", title="bin", title_fontsize=7)
    ax.axhline(0, color="k", linewidth=1)
    ax.set_ylabel("rate(bin) / rate(period) - 1")
    head = f"{feature} — " if feature else ""
    tail = f"consistency = {score:.2f}" if score is not None and not np.isnan(score) else ""
    ax.set_title(f"{head}temporal consistency {tail}".strip())
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
