"""Feature selection: optbinning IV/WoE, hard (rule-based), temporal stability, soft (voting) and shadow selection, interaction rules, importance."""

from train_model.feature_selection.binning import (
    WoETransformer,
    binning_iv,
    compute_iv,
    fit_optbinning,
)
from train_model.feature_selection.hard import HardFeatureSelector
from train_model.feature_selection.importance import ImportanceSelector, select_by_importance
from train_model.feature_selection.rules import Condition, Rule, TreeRuleMiner, extract_rules
from train_model.feature_selection.shadow import ShadowFeatureSelector
from train_model.feature_selection.soft import (
    ALL_CRITERIA,
    SoftFeatureSelector,
    best_k,
    forward_selection_gini,
)
from train_model.feature_selection.stability import (
    ConsistencyResult,
    TemporalStabilitySelector,
    temporal_consistency,
)

__all__ = [
    "Condition",
    "Rule",
    "ShadowFeatureSelector",
    "TreeRuleMiner",
    "extract_rules",
    "ConsistencyResult",
    "TemporalStabilitySelector",
    "temporal_consistency",
    "ALL_CRITERIA",
    "ImportanceSelector",
    "SoftFeatureSelector",
    "best_k",
    "forward_selection_gini",
    "select_by_importance",
    "HardFeatureSelector",
    "WoETransformer",
    "binning_iv",
    "compute_iv",
    "fit_optbinning",
]
