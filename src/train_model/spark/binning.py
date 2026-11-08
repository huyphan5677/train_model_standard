"""WoE transform on Spark from fitted optbinning objects (SQL CASE WHEN expressions).

:func:`woe_bins` (pure Python, testable without Spark) extracts the bin edges / categories and WoE values;
:func:`woe_case_when_expr` turns them into a Spark SQL expression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from optbinning import OptimalBinning
    from pyspark.sql import DataFrame


def woe_bins(optb: OptimalBinning) -> dict[str, Any]:
    """Return ``{"dtype", "bins", "missing_woe"}`` for a fitted OptimalBinning.

    numerical:  bins = [(lower, upper, woe), ...] with left-closed intervals [lower, upper)
    categorical: bins = [([categories...], woe), ...]
    """
    table = optb.binning_table.build()
    regular = table.loc[~table["Bin"].astype(str).isin(["Special", "Missing", ""]) & (table.index != "Totals")]
    woes = regular["WoE"].astype(float).tolist()
    missing = table.loc[table["Bin"].astype(str) == "Missing", "WoE"]
    missing_woe = float(missing.iloc[0]) if len(missing) else 0.0

    if optb.dtype == "numerical":
        edges = [-np.inf, *[float(s) for s in optb.splits], np.inf]
        bins = [(edges[i], edges[i + 1], woes[i]) for i in range(len(edges) - 1)]
    else:
        bins = [([str(c) for c in cats], woes[i]) for i, cats in enumerate(optb.splits)]
    return {"dtype": optb.dtype, "bins": bins, "missing_woe": missing_woe}


def apply_woe_bins(values, spec: dict[str, Any], handle_null: bool = True) -> np.ndarray:
    """Pure-numpy evaluation of :func:`woe_bins` (reference implementation mirroring the SQL)."""
    out = np.zeros(len(values), dtype=float)
    vals = list(values)
    for i, v in enumerate(vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[i] = spec["missing_woe"] if handle_null else 0.0
            continue
        if spec["dtype"] == "numerical":
            for lo, hi, w in spec["bins"]:
                if lo <= float(v) < hi:
                    out[i] = w
                    break
        else:
            for cats, w in spec["bins"]:
                if str(v) in cats:
                    out[i] = w
                    break
    return out


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def woe_case_when_expr(column: str, spec: dict[str, Any], handle_null: bool = True) -> str:
    """Spark SQL ``CASE WHEN`` for one feature. Values outside every bin (unseen categories) -> 0."""
    col = f"`{column}`"
    parts = [f"WHEN {col} IS NULL THEN {spec['missing_woe'] if handle_null else 0.0!r}"]
    if spec["dtype"] == "numerical":
        for lo, hi, w in spec["bins"]:
            conds = []
            if np.isfinite(lo):
                conds.append(f"{col} >= {lo!r}")
            if np.isfinite(hi):
                conds.append(f"{col} < {hi!r}")
            parts.append(f"WHEN {' AND '.join(conds) or 'TRUE'} THEN {w!r}")
    else:
        for cats, w in spec["bins"]:
            parts.append(f"WHEN CAST({col} AS STRING) IN ({', '.join(_sql_literal(c) for c in cats)}) THEN {w!r}")
    return "CASE " + " ".join(parts) + " ELSE 0.0 END"


def woe_transform_spark(df: DataFrame, binnings: dict[str, OptimalBinning], handle_null: bool = True) -> DataFrame:
    """Replace each binned column of ``df`` by its WoE value."""
    from pyspark.sql import functions as F

    for feature, optb in binnings.items():
        if feature in df.columns:
            expr = woe_case_when_expr(feature, woe_bins(optb), handle_null)
            df = df.withColumn(feature, F.expr(expr).cast("double"))
    return df
