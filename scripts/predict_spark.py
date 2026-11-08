"""Batch prediction job (spark-submit).

Example::

    spark-submit scripts/predict_spark.py \
        --config configs/model_config.yaml \
        --model-dir s3a://<bucket>/models/demo_cs_v1 \
        --input  s3a://<bucket>/features/merged \
        --output s3a://<bucket>/predict/demo_cs_v1 \
        --date 20250801

Reads ``{input}[/date={date}]`` parquet, scores it with the ModelBundle, writes
``{output}/predict[/date={date}]`` and ``{output}/shap_value[/date={date}]``.
S3 credentials come from env vars S3_ENDPOINT / S3_ACCESS_KEY / S3_SECRET_KEY (optional).

Note: ``--model-dir`` must be a local path (copy the bundle next to the job, or mount it).
"""

from __future__ import annotations

import argparse
import logging
import sys

from train_model.config import load_config
from train_model.modeling.bundle import ModelBundle
from train_model.spark.predict import predict_spark
from train_model.spark.scoring import proba_to_score_spark
from train_model.spark.session import create_spark_session, s3a_conf_from_env
from train_model.spark.shap import top_features_shap_spark

logger = logging.getLogger("predict_spark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="model YAML config")
    p.add_argument("--model-dir", required=True, help="local directory of the saved ModelBundle")
    p.add_argument("--input", help="input parquet path (default: config paths.predict_input)")
    p.add_argument("--output", help="output base path (default: config paths.predict_output)")
    p.add_argument("--date", help="snapshot date yyyymmdd -> reads/writes the date=<date> partition")
    p.add_argument("--id-col", help="id column (default: config model.id_col)")
    p.add_argument("--shap-top-n", type=int, default=10)
    p.add_argument("--no-shap", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cfg = load_config(args.config)
    bundle = ModelBundle.load(args.model_dir)

    input_path = args.input or cfg.paths["predict_input"]
    output_path = args.output or cfg.paths["predict_output"]
    suffix = f"/date={args.date}" if args.date else ""
    id_col = args.id_col or cfg.model.id_col

    spark_cfg = cfg.spark
    conf = {**(spark_cfg.get("conf") or {}), **s3a_conf_from_env()}
    spark = create_spark_session(spark_cfg.get("app_name", "train_model_predict"), spark_cfg.get("master"), conf)

    df = spark.read.parquet(input_path + suffix)
    logger.info("model=%s input=%s%s", bundle.model_code, input_path, suffix)

    scoring = cfg.scoring
    pred = predict_spark(spark, bundle, df, id_col=id_col)
    pred = proba_to_score_spark(
        pred,
        target_label=scoring.get("target_label", "good"),
        lower=scoring.get("lower", 300),
        upper=scoring.get("upper", 850),
        pct_lb=scoring.get("pct_lb", 0.01),
        pct_ub=scoring.get("pct_ub", 0.99),
        exponent=scoring.get("exponent", 1.3),
        bounds=bundle.score_bounds,
    )
    pred.write.mode("overwrite").parquet(f"{output_path}/predict{suffix}")

    if not args.no_shap:
        shap_df = top_features_shap_spark(spark, bundle, df, top_n=args.shap_top_n)
        shap_df.write.mode("overwrite").parquet(f"{output_path}/shap_value{suffix}")

    logger.info("done -> %s", output_path)
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
