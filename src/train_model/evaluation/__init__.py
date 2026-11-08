"""Model evaluation: AUC/Gini, decile table, PSI, SHAP, plots."""

from train_model.evaluation.metrics import auc_gini, auc_gini_by_group, decile_table, pr_auc_by_bin
from train_model.evaluation.plots import plot_forward_gini, plot_roc, plot_roc_by_group, plot_shap_summary
from train_model.evaluation.psi import psi, psi_by_bucket, psi_by_feature, psi_categorical
from train_model.evaluation.shap_utils import shap_group_importance, shap_importance

__all__ = [
    "auc_gini",
    "auc_gini_by_group",
    "decile_table",
    "plot_forward_gini",
    "plot_roc",
    "plot_roc_by_group",
    "plot_shap_summary",
    "pr_auc_by_bin",
    "psi",
    "psi_by_bucket",
    "psi_by_feature",
    "psi_categorical",
    "shap_group_importance",
    "shap_importance",
]
