"""Dataset helpers: synthetic demo data, time-based split, label summary."""

from train_model.data.label_analysis import label_summary
from train_model.data.split import time_split
from train_model.data.synthetic import INFORMATIVE_FEATURES, make_synthetic_dataset

__all__ = ["INFORMATIVE_FEATURES", "label_summary", "make_synthetic_dataset", "time_split"]
