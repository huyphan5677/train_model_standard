# train_model_standard

Bộ khung chuẩn để **train model phân loại nhị phân** (credit score, propensity, churn...) cho Data Team.
Gom và viết lại các hàm hay nhất từ `model_src` và repo `airflow-dags`. Code chung, không gắn khách hàng nào.

> **Cách dùng:** mỗi model mới → **clone repo này**, sửa `configs/model_config.yaml`, thay cell load data trong
> `notebooks/01_train.ipynb`, chạy từ trên xuống → có model + báo cáo đánh giá.

## Luồng train

```
data ─► label summary ─► time split ─► fill missing ─► HARD selection ─► SOFT selection (bỏ phiếu)
     ─► forward selection (chọn số feature) ─► train CatBoost ─► evaluate (Gini / decile / PSI / SHAP)
     ─► quy đổi điểm 300–850 ─► lưu ModelBundle ─► (Spark) predict hàng loạt
```

## Cài đặt

Cần [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv sync                    # thư viện train + dev (pytest, ruff, jupyter)
uv sync --extra spark      # thêm pyspark nếu máy có Spark / Java (chỉ cần cho predict)
```

Chạy notebook: mở `notebooks/01_train.ipynb` trong VS Code / Jupyter, chọn kernel `.venv`.

## Cấu trúc

```
configs/model_config.yaml     mọi tham số của 1 model (target, cột cat, split, ngưỡng selection, CatBoost, thang điểm, Spark)
notebooks/
  01_train.ipynb              train end-to-end trên data giả lập (đã chạy sẵn, xem output mẫu)
  02_predict_spark.ipynb      predict bằng Spark từ model đã lưu (chạy trên môi trường có Spark)
scripts/predict_spark.py      job spark-submit cho predict hàng loạt
src/train_model/
  config.py                   load_config
  data/                       make_synthetic_dataset, time_split, label_summary
  preprocessing/              fill_missing, encoder/imputer/scaler/transform kiểu sklearn, winsorizer
  feature_selection/          HardFeatureSelector, SoftFeatureSelector, forward_selection_gini,
                              select_by_importance, compute_iv, WoETransformer
  modeling/                   train_catboost, evaluate_splits, ModelBundle,
                              CatBoostEnsembleClassifier / TransferLearner / Shrinker
  evaluation/                 auc_gini(_by_group), decile_table, psi(_by_feature/_by_bucket), shap_importance, plots
  scoring/                    fit_score_bounds, proba_to_score
  spark/                      create_spark_session, predict_spark, proba_to_score_spark,
                              woe_transform_spark, top_features_shap_spark   (cần pyspark)
tests/                        pytest trên data giả lập
```

## Feature selection

### Hard selection — `HardFeatureSelector`

Loại theo luật, lần lượt. Kết quả ở `report_` (cột nào bị loại, ở bước nào, giá trị bao nhiêu):

| Bước | Loại khi | Config |
|---|---|---|
| `missing` | tỉ lệ missing > ngưỡng | `missing_threshold` (0.95) |
| `constant` | một giá trị chiếm ≥ ngưỡng | `quasi_constant_tol` (0.98) |
| `duplicate` | trùng hoàn toàn cột khác | — |
| `low_iv` | Information Value < ngưỡng | `iv_threshold` (0.02) |
| `high_corr` | cặp \|corr\| > ngưỡng → bỏ cột IV thấp hơn | `corr_threshold` (0.9) |

`transform` trả về **giá trị gốc** của các cột được giữ. Muốn đổi sang WoE, dùng thêm `WoETransformer`.

### Soft selection — `SoftFeatureSelector` (bỏ phiếu)

10 tiêu chí, mỗi tiêu chí cho feature một phiếu. Giữ feature có **≥ `min_votes`** phiếu.
Bảng phiếu đầy đủ ở `votes_`.

| Tiêu chí | Được phiếu khi |
|---|---|
| `correlation` | nằm trong top-k \|tương quan\| với label |
| `permutation` | nằm trong top-k AUC giảm khi xáo trộn feature (trên validation) |
| `single_feature` | nằm trong top-k AUC khi train chỉ bằng feature đó |
| `tree_importance` | nằm trong top-k importance của RandomForest |
| `tree_vs_random` | importance cao hơn một cột ngẫu nhiên |
| `variance` | variance > `variance_threshold` |
| `mutual_info` | nằm trong top `mi_percentile`% mutual information |
| `l1` / `l2` | được Logistic Regression L1 / L2 giữ lại |
| `rfe` | được RFE (DecisionTree) chọn |

Sau đó `forward_selection_gini` thêm dần feature theo thứ tự số phiếu và đo Gini trên validation.
`best_k` trả về số feature ít nhất mà Gini chỉ kém mức cao nhất ≤ 0.005.

## Model đã lưu — `ModelBundle`

```
artifacts/<model_code>/model.cbm   # CatBoost
artifacts/<model_code>/meta.json   # features, cột cat, cách fill missing, score bounds, metrics, version
```

`ModelBundle.load(dir).predict_proba(df_raw)` tự áp dụng đúng cách xử lý missing như lúc train.

**Thang điểm:** `fit_score_bounds` tính khoảng điểm trên train và lưu vào bundle. Nhờ vậy điểm các tháng sau
dùng chung một thang, không bị kéo giãn theo từng batch như cách cũ (cách cũ vẫn dùng được nếu không truyền `bounds`).

## Predict trên Spark

> Code Spark được viết cho môi trường có Spark. Repo **không chạy** phần này trong test local:
> `tests/test_spark.py` tự bỏ qua khi không có pyspark. Lần đầu dùng trên cluster, nên chạy
> `pytest tests/test_spark.py` để kiểm tra.

```bash
export S3_ENDPOINT=... S3_ACCESS_KEY=... S3_SECRET_KEY=...   # nếu đọc/ghi s3a
spark-submit scripts/predict_spark.py \
  --config configs/model_config.yaml \
  --model-dir artifacts/demo_cs_v1 \
  --input  s3a://<bucket>/features/merged \
  --output s3a://<bucket>/predict/demo_cs_v1 \
  --date 20250801
```

Kết quả: `predict/date=...` (id, `predict_proba`, `score`) và `shap_value/date=...` (top feature SHAP).
Cấu hình cluster (k8s image, memory...) đặt trong `spark.conf` của config. Không hard-code trong code.

## Làm model mới — checklist

1. Clone repo, tạo branch.
2. Copy `configs/model_config.yaml` → sửa `model_code`, `target_col`, `cat_cols`, `exclude_cols`, `split_date`.
3. Trong `01_train.ipynb`, thay cell **Load data** bằng data thật và đổi đường dẫn config ở cell Setup nếu cần.
4. Chạy notebook, xem `report_` (hard), `votes_` (soft), Gini train/val/test, decile, PSI, SHAP.
5. Model nằm ở `artifacts/<model_code>/`. Đưa thư mục này lên nơi lưu model của team để chạy predict.

## Dev

```bash
uv run pytest            # test (test Spark tự bỏ qua nếu không có pyspark)
uv run ruff check .      # lint
```

Output đang commit trong `01_train.ipynb` là kết quả chạy trên data giả lập (seed 42).
