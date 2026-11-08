"""Spark batch prediction helpers (requires ``pip install train-model[spark]``).

Import the sub-modules explicitly, e.g. ``from train_model.spark.predict import predict_spark``.
Nothing here is imported by the rest of the package, so training works without pyspark.

NOTE: this code is written for the team's Spark environment and is not executed in local CI
(``tests/test_spark.py`` is skipped when pyspark is unavailable).
"""
