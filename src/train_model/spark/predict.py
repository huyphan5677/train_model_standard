"""Batch prediction on Spark with a broadcast CatBoost model inside a pandas UDF."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd

from train_model.preprocessing.fill import to_category_str

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from train_model.modeling.bundle import ModelBundle

_NULL_TOKENS = ("", "nan", "none", "null")


def fill_missing_spark(
    df: DataFrame,
    cat_cols: Sequence[str],
    cat_fill: str = "unknown",
    num_fill: float | None = None,
) -> DataFrame:
    """Spark twin of :func:`train_model.preprocessing.fill_missing` (same rules)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType

    for c in cat_cols:
        if c in df.columns:
            v = F.lower(F.trim(F.col(c).cast("string")))
            df = df.withColumn(c, F.when(v.isNull() | v.isin(*_NULL_TOKENS), F.lit(cat_fill)).otherwise(v))
    if num_fill is not None:
        num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
        df = df.fillna(num_fill, subset=num_cols)
    return df


def predict_spark(
    spark: SparkSession,
    bundle: ModelBundle,
    df: DataFrame,
    id_col: str,
    keep_cols: Sequence[str] = (),
    return_features: bool = False,
    proba_col: str = "predict_proba",
    repartition_factor: int | None = 3,
) -> DataFrame:
    """Score ``df`` with ``bundle.model``. Returns ``id_col, *keep_cols, proba_col`` (+ features if asked).

    Preprocessing = the bundle's training-time fill rules; feature order = ``bundle.features``.
    """
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    missing = [c for c in bundle.features if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing model features: {missing}")

    if repartition_factor:
        df = df.repartition(spark.sparkContext.defaultParallelism * repartition_factor)
    df = fill_missing_spark(df, bundle.cat_features, bundle.cat_fill, bundle.num_fill)

    features = list(bundle.features)
    cat_set = set(bundle.cat_features)
    model_bc = spark.sparkContext.broadcast(bundle.model)

    @F.pandas_udf(returnType=T.DoubleType())
    def _predict(*cols):
        X = pd.concat([pd.Series(c) for c in cols], axis=1)
        X.columns = features
        for c in cat_set:
            X[c] = to_category_str(X[c])
        return pd.Series(model_bc.value.predict_proba(X)[:, 1])

    out = df.withColumn(proba_col, _predict(*[F.col(c) for c in features]))
    cols = [id_col, *keep_cols, proba_col] + (features if return_features else [])
    return out.select(*cols)
