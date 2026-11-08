# train_model_standard

Bộ khung train model phân loại nhị phân (credit score, propensity, churn...). Mỗi model mới: clone repo,
sửa `configs/model_config.yaml`, thay cell load data trong `notebooks/01_train.ipynb`, chạy từ trên xuống.

## Cài đặt

```bash
uv venv --python 3.11
uv sync                    # train + dev (pytest, ruff, jupyter)
uv sync --extra spark      # thêm pyspark (chỉ cần cho predict)
```

## Luồng train (`notebooks/01_train.ipynb`)

```
data ─► label summary ─► time split ─► fill missing ─► hard selection ─► temporal stability
     ─► soft selection ─► shadow check ─► forward selection ─► CatBoost
     ─► evaluate (Gini/KS/decile/PSI/calibration/SHAP) ─► drift (event PSI, SHAP drift)
     ─► challenger scorecard (+ interaction rules) + compare_models (DeLong) ─► ablation
     ─► score 300–850 ─► ModelBundle ─► (Spark) predict
```

Test set = các tháng `>= split_date`, chỉ dùng để đánh giá — không dùng để chọn feature hay model.

## Cấu trúc

```
configs/model_config.yaml   tham số của 1 model
notebooks/                  01_train (đã chạy trên data giả lập), 02_predict_spark
scripts/predict_spark.py    job spark-submit
src/train_model/
  data/                     make_synthetic_dataset, time_split, label_summary
  preprocessing/            fill_missing, encoder / imputer / scaler / winsorizer
  feature_selection/        HardFeatureSelector, TemporalStabilitySelector, SoftFeatureSelector,
                            ShadowFeatureSelector, forward_selection_gini, TreeRuleMiner, WoETransformer
  modeling/                 train_catboost, ModelBundle, LogisticScorecard, CatBoost utils
  evaluation/               auc_gini, ks_statistic, delong_test, decile_table, psi / psi_report,
                            calibration_*, cross_audit, compare_models, *_ablation, shap_*, plots
  scoring/                  fit_score_bounds, proba_to_score
  spark/                    predict_spark, proba_to_score_spark, woe_transform_spark, shap top
tests/                      pytest trên data giả lập
```

Cách dùng và phương pháp của từng hàm nằm trong docstring. Một số bước (temporal stability, shadow check,
calibration, cross-audit, compare_models, DeLong, event PSI, SHAP drift, interaction rules, ablation) lấy ý tưởng từ
[AIDflow](https://github.com/aigensys-labs/aidflow) và được viết lại.

## Predict trên Spark

Cài package trên cluster (`pip install ".[spark]"` hoặc wheel qua `--py-files`), rồi:

```bash
spark-submit scripts/predict_spark.py --config configs/model_config.yaml \
  --model-dir artifacts/<model_code> --input s3a://<bucket>/features --output s3a://<bucket>/predict --date 20250801
```

Key S3 lấy từ `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`. Phần Spark không chạy trong test local.

## Dev

```bash
uv run pytest
uv run ruff check .
```
