"""Model evaluation: AUC/Gini/KS, decile table, PSI, calibration, cross-audit, model comparison, SHAP, plots."""

from train_model.evaluation.calibration import calibration_summary, calibration_table
from train_model.evaluation.comparison import ComparisonResult, compare_models, feature_exposure
from train_model.evaluation.cross_audit import CrossAuditResult, cross_audit
from train_model.evaluation.metrics import (
    auc_gini,
    auc_gini_by_group,
    decile_table,
    ks_statistic,
    pr_auc_by_bin,
)
from train_model.evaluation.plots import (
    plot_calibration,
    plot_forward_gini,
    plot_roc,
    plot_roc_by_group,
    plot_shap_summary,
    plot_temporal_consistency,
)
from train_model.evaluation.psi import psi, psi_by_bucket, psi_by_feature, psi_categorical
from train_model.evaluation.shap_utils import shap_group_importance, shap_importance

__all__ = [
    "ComparisonResult",
    "CrossAuditResult",
    "calibration_summary",
    "calibration_table",
    "compare_models",
    "cross_audit",
    "feature_exposure",
    "ks_statistic",
    "plot_calibration",
    "plot_temporal_consistency",
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
