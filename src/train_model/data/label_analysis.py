"""Label rate summaries (overall / per month / per group)."""

from __future__ import annotations

import pandas as pd


def _rates(frame: pd.DataFrame, label: str, by: str | None) -> pd.DataFrame:
    if by is None:
        out = pd.DataFrame({"count": [frame[label].count()], "n_positive": [frame[label].sum()]})
    else:
        out = frame.groupby(by)[label].agg(count="count", n_positive="sum").reset_index()
    out["rate"] = out["n_positive"] / out["count"]
    return out


def label_summary(
    df: pd.DataFrame,
    label: str,
    date_col: str,
    group_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return ``{"overall", "by_month", "by_group"}`` tables with ``count, n_positive, rate``.

    ``by_group`` is only present when ``group_col`` is given.
    """
    result = {
        "overall": _rates(df, label, None),
        "by_month": _rates(df, label, date_col),
    }
    if group_col is not None:
        result["by_group"] = _rates(df, label, group_col)
    return result
