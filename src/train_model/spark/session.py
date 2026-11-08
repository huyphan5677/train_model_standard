"""SparkSession creation. All cluster / storage settings come from config or environment — nothing hard-coded."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


def s3a_conf_from_env(
    endpoint_var: str = "S3_ENDPOINT",
    key_var: str = "S3_ACCESS_KEY",
    secret_var: str = "S3_SECRET_KEY",
    ssl: bool = False,
) -> dict[str, str]:
    """Spark ``fs.s3a.*`` settings built from environment variables (empty dict if any is missing)."""
    endpoint, key, secret = (os.getenv(v) for v in (endpoint_var, key_var, secret_var))
    if not (endpoint and key and secret):
        return {}
    return {
        "spark.hadoop.fs.s3a.endpoint": endpoint,
        "spark.hadoop.fs.s3a.access.key": key,
        "spark.hadoop.fs.s3a.secret.key": secret,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": str(ssl).lower(),
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    }


def create_spark_session(
    app_name: str,
    master: str | None = "local[*]",
    conf: dict[str, str] | None = None,
    log_level: str = "ERROR",
) -> SparkSession:
    """Build (or get) a SparkSession.

    ``master=None`` lets ``spark-submit`` / the cluster decide. ``conf`` is any extra Spark config
    (k8s image, executor memory, event log dir, s3a settings from :func:`s3a_conf_from_env`, ...).
    """
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    for key, value in (conf or {}).items():
        builder = builder.config(key, str(value))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)

    logger.info("Spark %s | app=%s | id=%s", spark.version, app_name, spark.sparkContext.applicationId)
    for key in ("spark.driver.memory", "spark.executor.memory", "spark.executor.cores", "spark.executor.instances"):
        logger.info("%s = %s", key, spark.conf.get(key, "not set"))
    return spark
