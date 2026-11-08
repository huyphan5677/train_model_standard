"""Train / validation / test split by snapshot date (out-of-time test set)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def time_split(
    df: pd.DataFrame,
    date_col: str,
    split_date: int | dict[str, int],
    val_size: float = 0.2,
    seed: int = 42,
    group_col: str | None = None,
    target_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (train, val, test).

    * test  = rows with ``date_col >= split_date`` (out-of-time).
    * val   = random ``val_size`` share of the remaining rows (stratified on ``target_col`` if given).
    * ``split_date`` may be a dict ``{group_value: date}`` for a per-client cut-off; then
      ``group_col`` is required. Groups missing from the dict go entirely to train/val.

    The input frame is never modified.
    """
    dates = df[date_col].astype(int)
    if isinstance(split_date, dict):
        if group_col is None:
            raise ValueError("split_date given as dict requires group_col")
        cutoff = df[group_col].map(split_date)
        is_test = cutoff.notna() & (dates >= cutoff.fillna(0))
    else:
        is_test = dates >= int(split_date)

    test = df.loc[is_test].reset_index(drop=True)
    dev = df.loc[~is_test]
    if len(dev) == 0:
        raise ValueError("No rows before split_date - nothing left for training")

    stratify = dev[target_col] if target_col is not None and dev[target_col].nunique() > 1 else None
    train, val = train_test_split(dev, test_size=val_size, random_state=seed, stratify=stratify)
    return train.reset_index(drop=True), val.reset_index(drop=True), test
