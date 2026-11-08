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
