"""Top-N SHAP importance computed on a driver-side sample of a Spark DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from train_model.evaluation.shap_utils import shap_importance

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

    from train_model.modeling.bundle import ModelBundle


def top_features_shap_spark(
    spark: SparkSession,
    bundle: ModelBundle,
    df: DataFrame,
    top_n: int = 10,
    sample_size: int = 50_000,
    seed: int = 42,
) -> DataFrame:
    """Mean |SHAP| of the ``top_n`` features. Returns ``DataFrame[model_code, feature_name, rank, shap_value]``."""
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    total = df.count()
    fraction = min(1.0, 1.2 * sample_size / total) if total else 1.0
    pdf = df.select(*bundle.features).sample(fraction=fraction, seed=seed).limit(sample_size).toPandas()
    imp = shap_importance(bundle.model, bundle.preprocess(pdf), max_samples=sample_size).head(top_n)

    rows = [
        (bundle.model_code, str(r.feature), rank, float(r.mean_abs_shap))
        for rank, r in enumerate(imp.itertuples(index=False), start=1)
    ]
    schema = StructType(
        [
            StructField("model_code", StringType(), True),
            StructField("feature_name", StringType(), True),
            StructField("rank", IntegerType(), True),
            StructField("shap_value", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(rows, schema=schema)
