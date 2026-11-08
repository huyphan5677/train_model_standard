# train_model_standard — Thiết kế

Ngày: 2026-11-08 · Remote: https://github.com/huyphan5677/train_model_standard.git · Vị trí local: `C:\Users\iltv1\Desktop\train_model_standard`

## 1. Mục đích

Team đang có code train model rải ở 2 nơi:

- `ML_PLATFORM/model_src` — module rời: hard/soft/group feature selection, transformer, xử lý missing/outlier, evaluation, tiện ích CatBoost (ensemble, transfer learning, shrink), label analysis, data checker.
- `Desktop/airflow-dags` (repo Unitel) — pipeline train CatBoost (`src/models/train.py`), predict Spark (`predict_base.py`), monitor PSI (`monitor_base.py`), tiện ích Spark/S3 (`src/utils/common.py`).

Code đang trùng lặp (PSI có 3 bản, `select_important_features` 2 bản), hard-code theo khách hàng (cột cat Unitel, `s3a://unitel/`, image k8s), dùng biến global, và có bug. Mục tiêu: **một repo khung chuẩn kế thừa phần tốt nhất của cả hai**, để khi làm model mới, thành viên team **clone nguyên repo**, sửa config, chạy notebook là ra model.

## 2. Mục tiêu / Không làm / Không được xảy ra

**Mục tiêu**
- Gom các hàm hay của 2 nguồn thành package `train_model` gọn, mỗi chức năng một bản duy nhất, không gắn khách hàng.
- `notebooks/01_train.ipynb` chạy end-to-end trên **data giả lập** (ai clone cũng chạy được): sinh data → hard selection → soft selection (bỏ phiếu) → train CatBoost → evaluate → lưu model.
- `notebooks/02_predict_spark.ipynb` + `scripts/predict_spark.py`: code predict bằng Spark (pandas_udf + broadcast model) → điểm → SHAP top. Viết sẵn cho môi trường có Spark, không chạy local.
- Đẩy lên GitHub remote ở trên.

**Không làm** (người dùng xác nhận)
- Feature engineering theo nhà mạng (`src/features/*_lxm.py`).
- DAG Airflow, data quality, job quality (`data_checker.py`, `src/data_quality`, `src/job_quality`, `JobQuality`).
- Ghi MySQL (`insert_score.py`), Telegram.
- **Chạy phần Spark trên máy local.** Code Spark (predict, score, SHAP) vẫn có trong repo, nhưng máy này không có Spark nên không chạy/verify; chỉ notebook train được chạy (người dùng chốt 2026-11-08).

**Không được xảy ra**
- Notebook train không chạy được trên máy mới clone (cần S3/cluster/data thật).
- Còn hard-code tên khách hàng / bucket / cột cụ thể trong `src/`.
- Hàm sửa ngầm DataFrame của người gọi.

## 3. Ràng buộc

- Máy dev hiện tại: Python hệ thống 3.9, có `uv` 0.9, **không có Spark** → `notebooks/02_predict_spark.ipynb` và `scripts/predict_spark.py` **được viết nhưng không chạy**; `pyspark` là extra tuỳ chọn, không cài vào môi trường local; test Spark tự skip (`pytest.importorskip("pyspark")`). Code Spark chỉ được kiểm tra bằng lint + import-safe (module `train_model.spark` không bị import khi dùng phần train).
- `requires-python = ">=3.10"`; dev bằng `uv` với Python 3.11.
- Dependency lõi: pandas, numpy, scikit-learn, catboost, optbinning, feature-engine, shap, matplotlib, seaborn, pyyaml. Extra tuỳ chọn: `spark` (pyspark), `s3` (s3fs, boto3). Dev: ruff, pytest, nbconvert/ipykernel.

## 4. Cấu trúc repo

```
train_model_standard/
├── configs/model_config.yaml
├── src/train_model/
│   ├── config.py               load + validate YAML → dataclass
│   ├── data/                   synthetic.py, split.py, label_analysis.py
│   ├── preprocessing/          imputers, outliers, encoders, transforms (sklearn-compatible)
│   ├── feature_selection/      hard.py, soft.py, importance.py, binning.py
│   ├── modeling/               train.py, bundle.py, catboost_utils.py
│   ├── evaluation/             metrics.py, psi.py, shap_utils.py, plots.py
│   ├── scoring/                score.py (pandas)
│   ├── spark/                  session.py, predict.py, binning.py, scoring.py, shap.py
│   └── io.py                   save/load model bundle (local hoặc s3 qua fsspec)
├── notebooks/01_train.ipynb, 02_predict_spark.ipynb
├── scripts/predict_spark.py
├── tests/
├── pyproject.toml, README.md, .gitignore, .ruff.toml
└── docs/superpowers/specs/ (file này)
```

## 5. Hành vi từng phần

### 5.1 Config (`configs/model_config.yaml`)
Một file / model. Các khối: `model` (model_code, target_col, id_col, date_col, group_col), `features` (cat_cols, num_cols hoặc auto-detect, exclude_cols, cat_fill / num_fill), `split` (split_date, val_size, seed), `hard_selection` (missing_threshold, iv_threshold, corr_threshold, quasi_constant_tol), `soft_selection` (criteria bật/tắt, top_k, min_votes), `catboost` (params), `scoring` (score range 300–850, pct clip, exponent, target_label good/bad), `spark` (app_name, master, extra conf dict), `paths` (artifacts_dir, input/output predict). Không chứa secret — key S3 đọc từ biến môi trường.

### 5.2 Data
- `make_synthetic_dataset(n_rows, n_months, seed)`: data kiểu telco có prefix nhóm (`usage_`, `recharge_`, `voice_`, `sms_`, `device_`, `demographic_`), cột `date` (yyyymmdd theo tháng), `client`, vài cột cat, cột có missing nhiều, cột hằng, cột trùng, cặp cột tương quan cao, label `GB` sinh từ một tập feature "thật" → để chứng minh selection lọc đúng.
- `time_split(df, date_col, split_date, val_size, seed)` → train/val/test (test = sau split_date, val random từ train). Hỗ trợ `split_date` dạng dict theo client (kế thừa `split_train_test_val`).
- `label_summary(df, label, date_col, group_col)`: tỉ lệ label tổng / theo tháng / theo client (rút gọn từ `label_analysis.py`, chỉ pandas).

### 5.3 Preprocessing
Giữ các transformer sklearn-compatible của `transform.py` (Log, Power, QuantileBinner, Scaler, LabelEncoder, OneHot, Frequency, Target encoder, Mean/Median/Mode/Constant imputer) và `missing_ouliers_processing.py` (step builder cho feature-engine imputer/Winsorizer, `CapCategoricalOutlier`). Sửa bug typo `CategoricaCategoricalImputerlImputer`. Thêm `fill_missing(df, cat_fill, num_fill)` dùng chung cho train và predict (thay cho `preprocess_for_prediction` hard-code; fill strategy được lưu vào bundle): chuẩn hoá cat (trim, lower, rỗng → giá trị fill), num fill theo config.

### 5.4 Feature selection
- **binning.py**: `fit_optbinning`, `woe_transform`, `compute_iv(df, features, target)` — một bản duy nhất dựa trên `optbinning`, tự nhận numerical/categorical.
- **hard.py** — `HardFeatureSelector(BaseEstimator, TransformerMixin)`: các bước MissingRatio → Constant/Quasi-constant → Duplicate → IV thấp → Tương quan cao (giữ cột IV cao hơn, xử lý cặp theo |corr| giảm dần). Mỗi bước lưu lý do loại vào `report_` (DataFrame: feature, step, value). IV lưu trong `iv_` của instance, **không global**. `transform` trả về **cột gốc đã chọn** (không đổi giá trị); WoE là bước riêng tuỳ chọn (`WoETransformer`).
- **soft.py** — `SoftFeatureSelector`: 10 tiêu chí như bản cũ (correlation |r|, permutation, single-feature AUC, tree importance, tree vs random column, variance, mutual information, L1, L2, RFE), mỗi tiêu chí bật/tắt qua `criteria`. Xếp hạng dùng rank thật (sửa bug `argsort`), không mutate input, L1/L2 chạy trên dữ liệu đã chuẩn hoá. Output `votes_` (feature × tiêu chí Y/N + `n_votes`), `selected_` = features có `n_votes >= min_votes`. Cột cat: tiêu chí cần số sẽ dùng bản encode nội bộ (ordinal) — không ảnh hưởng output.
- `forward_selection_gini(X_train, y_train, X_val, y_val, ordered_features, step)`: thay `CheckPerformance` — thêm dần feature theo thứ tự phiếu, trả bảng Gini + vẽ, `best_k(tolerance=0.005)` trả **số feature** (sửa bug cũ).
- **importance.py** — `select_by_importance(X, y, estimator, top_k|threshold|accumulate)` + `ImportanceSelector` transformer. Chuẩn hoá importance về tổng 1 trước khi tính accumulate (sửa lỗi lệch giữa CatBoost tổng 100 và sklearn tổng 1).
- `GroupFeatureSelection` cũ = Hard + Importance → thay bằng sklearn `Pipeline` mẫu trong notebook, không giữ class riêng.

### 5.5 Modeling
- `train_catboost(train, val, features, cat_features, target, params, seed)` → model; không scale mặc định (CatBoost không cần), tuỳ chọn MinMax giữ lại như `scale_features`.
- `ModelBundle` (bundle.py): model CatBoost + metadata (features, cat_features, fill strategy, model_code, train date range, score bounds, metrics, version thư viện). Lưu thành thư mục `model.cbm` + `meta.json`; có hàm load. Predict Spark đọc bundle để tiền xử lý giống lúc train.
- catboost_utils.py: `CatBoostEnsembleClassifier`, `CatBoostTransferLearner`, `CatBoostShrinker` từ `models.py` (rà lại, giữ API).

### 5.6 Evaluation
- metrics.py: `auc_gini`, `auc_gini_by_group`, `decile_table` (từ `model_decile_summary`, bỏ tên cột Unitel, không cần cột score), `pr_auc_by_bin`.
- psi.py: **một** `psi(expected, actual, bins, method="quantile"|"uniform")` + `psi_safe` (NaN/rỗng) + `psi_by_feature(df_ref, df_cur, features)` + `psi_by_bucket`.
- shap_utils.py: `shap_importance`, `shap_top_n`, `shap_group_importance` (gom theo prefix).
- plots.py: ROC nhiều tập, ROC theo nhóm, SHAP summary.

### 5.7 Scoring
`proba_to_score` (pandas) và `spark/scoring.py` (Spark) cùng công thức của `transform_to_com_score`: đảo theo target_label → clip theo quantile → −logit → lũy thừa có dấu `exponent` → scale về [300, 850]. Thêm tham số `bounds`: nếu truyền (lấy từ bundle, tính trên train) thì scale cố định → điểm ổn định giữa các tháng; nếu không thì dùng min/max của batch như hành vi cũ.

### 5.8 Spark (viết, không chạy local)
- session.py: `create_spark_session(app_name, master="local[*]", conf: dict)` — mọi cấu hình k8s/S3 truyền qua dict từ config, không hard-code. Helper `s3a_conf_from_env()`.
- predict.py: `predict_spark(spark, bundle, df, id_col, return_features=False)` → pandas_udf + broadcast model, đúng thứ tự `feature_names_`, ép cat về string; `fill_missing_spark` dùng fill từ bundle.
- binning.py: `woe_case_when_expr`, `woe_transform_spark` (từ `binning_expr`/`opt_binning_num_transform`).
- shap.py: `top_features_shap_spark` (sample → pandas → TreeExplainer → Spark DF `model_code, feature_name, rank, shap_value`).
- `scripts/predict_spark.py --config --model-dir --input --output [--date]`: đọc parquet, predict, score, ghi parquet (local hoặc s3a). Không có JobQuality/Telegram/MySQL.

### 5.9 Notebook
- 01_train: theo đúng thứ tự mục tiêu, có markdown giải thích từng bước bằng tiếng Việt, in report hard selection, bảng phiếu soft selection, đường Gini forward, metric train/val/test, decile, PSI train vs test, SHAP, lưu bundle vào `artifacts/`, load lại để kiểm tra; tính `proba_to_score` pandas + score bounds lưu vào bundle. **Được chạy để verify** (nbconvert execute) trước khi commit.
- 02_predict_spark: tạo Spark session, đọc data, load bundle, predict, score, SHAP top, ghi parquet. **Không chạy trên máy này** — commit không kèm output.

### 5.10 Test & chất lượng
pytest trên data giả: hard selector loại đúng cột hằng/trùng/missing/tương quan; soft selector chọn được feature "thật", không mutate input, `min_votes` hoạt động; PSI (bằng 0 khi cùng phân phối, dương khi lệch); proba_to_score trong khoảng và đơn điệu; train + save/load bundle round-trip. ruff lint sạch.

### 5.11 Git
`git init` nhánh `main`, `.gitignore` loại `.venv`, `artifacts/`, `*.cbm`, `catboost_info/`, data. Commit và push lên remote. Máy không có `gh`; push bằng git HTTPS (credential manager của Windows).

## 6. Quyết định do builder (Claude) tự chọn — người dùng kiểm tra lại
- Tên package `train_model`; Python ≥3.10, dev 3.11 qua uv.
- Hard selector trả cột gốc (không WoE); WoE là bước riêng tuỳ chọn.
- Bỏ các hàm đang comment trong `train.py` (Boruta, MRMR, Optuna, time-series CV, Gini theo segment).
- Score scale cố định theo bounds từ train (có fallback hành vi cũ).
- Model lưu dạng `model.cbm` + `meta.json` thay vì pickle.
- Notebook train commit kèm output để team xem kết quả mẫu.
- `label_analysis` rút gọn thành `label_summary` pandas; `data_checker` bỏ (thuộc data quality).

## 7. Để builder tự quyết
Chi tiết sinh data giả, tên hàm nội bộ, cách tổ chức test, nội dung README.
