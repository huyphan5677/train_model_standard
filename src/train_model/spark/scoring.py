"""Probability -> score on Spark. Same formula as :func:`train_model.scoring.proba_to_score`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from train_model.scoring.score import _EPS, _label_int

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


def proba_to_score_spark(
    df: DataFrame,
    proba_col: str = "predict_proba",
    score_col: str = "score",
    target_label: Literal["good", "bad"] = "good",
    lower: int = 300,
    upper: int = 850,
    pct_lb: float = 0.01,
    pct_ub: float = 0.99,
    exponent: float = 1.3,
    bounds: tuple[float, float] | None = None,
    relative_error: float = 1e-6,
) -> DataFrame:
    """Add integer ``score_col`` in ``[lower, upper]``.

    Pass ``bounds=bundle.score_bounds`` (fit on training data) for a scale that is stable across runs;
    without it the batch's own quantiles / min / max are used (legacy behaviour).
    """
    from pyspark.sql import functions as F

    tmp = "__p_bad"
    df = df.withColumn(tmp, F.abs(F.lit(float(_label_int(target_label))) - F.col(proba_col)))
    if bounds is None:
        q_lo, q_hi = df.approxQuantile(tmp, [pct_lb, pct_ub], relative_error)
        df = df.withColumn(tmp, F.least(F.greatest(F.col(tmp), F.lit(q_lo)), F.lit(q_hi)))
    df = df.withColumn(tmp, F.least(F.greatest(F.col(tmp), F.lit(_EPS)), F.lit(1 - _EPS)))

    raw = -F.log(F.col(tmp) / (F.lit(1.0) - F.col(tmp)))
    df = df.withColumn(score_col, F.signum(raw) * F.pow(F.abs(raw), F.lit(exponent)))

    if bounds is None:
        lo, hi = df.agg(F.min(score_col), F.max(score_col)).first()
    else:
        lo, hi = bounds
    if hi is None or lo is None or hi <= lo:
        return df.withColumn(score_col, F.lit((lower + upper) // 2)).drop(tmp)

    factor = float(upper - lower) / float(hi - lo)
    scaled = F.floor((F.col(score_col) - F.lit(float(lo))) * F.lit(factor) + F.lit(0.5)) + F.lit(lower)
    return df.withColumn(score_col, F.least(F.greatest(scaled, F.lit(lower)), F.lit(upper)).cast("int")).drop(tmp)
