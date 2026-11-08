# train_model_standard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cloneable template repo whose package `train_model` merges the best training code from `model_src` and `airflow-dags`, with a runnable training notebook on synthetic data and (unexecuted) Spark predict code.

**Architecture:** `src/train_model/` is a pure-pandas/sklearn/CatBoost library split by stage (data → preprocessing → feature_selection → modeling → evaluation → scoring), plus an isolated `spark/` subpackage that nothing else imports. One YAML config per model drives the notebook. The trained model is persisted as a `ModelBundle` directory (`model.cbm` + `meta.json`) that the Spark code reads.

**Tech Stack:** Python ≥3.10 (dev 3.11 via uv), pandas, numpy, scikit-learn, catboost, optbinning, feature-engine, shap, matplotlib, seaborn, pyyaml; pytest, ruff, nbconvert/ipykernel; optional extras `spark` (pyspark), `s3` (s3fs, boto3).

**Spec:** `docs/superpowers/specs/2026-11-08-train-model-standard-design.md`

**Source code to port from** (read the relevant file before each task):
- `M` = `I:\My Drive\Tài liệu\BLUEINFO\DATA TEAM DOCUMENTS\ML_PLATFORM\model_src\`
- `A` = `C:\Users\iltv1\Desktop\airflow-dags\src\`

## Global Constraints

- Repo root: `C:\Users\iltv1\Desktop\train_model_standard`; package `src/train_model/`; remote `https://github.com/huyphan5677/train_model_standard.git`, branch `main`.
- `requires-python = ">=3.10"`; local env `uv venv --python 3.11`; `pyspark` NOT installed locally.
- No customer names, bucket names (`unitel`, `s3a://unitel/`), k8s image names or hard-coded column lists in `src/`.
- No module-level mutable globals; no function mutates a DataFrame passed in by the caller (copy first).
- Only one implementation each of PSI, IV/WoE, importance-based selection.
- `train_model/__init__.py` and every non-spark module must import without pyspark installed.
- Docstrings / comments in English; notebook markdown in Vietnamese.
- `ruff check .` clean (line-length 110, rules E,F,I,B,UP).

## Review Focus

1. Categorical columns with NaN / mixed types fed to selectors and CatBoost → must not crash (cast to str, fill with config value). Test in Task 4 (`test_hard_selector_handles_cat_with_nan`) and Task 7 (`test_train_with_cat_nan`).
2. A column that is all-NaN or constant → OptimalBinning fails; hard selector must drop it with a reason, not raise. Test in Task 4 (`test_hard_selector_drops_all_nan_column`).
3. PSI with empty input, all-equal values (duplicate quantile edges) → returns NaN / finite, never raises. Test in Task 6 (`test_psi_constant_and_empty`).
4. Decile table when predictions have many ties (`qcut` duplicates) → fewer bins, no crash. Test in Task 6 (`test_decile_table_with_ties`).
5. `ModelBundle.load` on a bundle saved by a different session → features/cat_features/fill values round-trip exactly and predictions match. Test in Task 7 (`test_bundle_roundtrip_predictions_equal`).

---

### Task 1: Scaffold, config loader, git init

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.ruff.toml`, `README.md` (stub), `configs/model_config.yaml`, `src/train_model/__init__.py`, `src/train_model/config.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> ModelConfig`; `ModelConfig` dataclass with attributes `model, features, split, hard_selection, soft_selection, catboost, scoring, spark, paths` (each a plain dict except `model`, `features`, `split` which are dataclasses below).
  - `ModelSection(model_code: str, target_col: str, id_col: str, date_col: str, group_col: str | None)`
  - `FeatureSection(cat_cols: list[str], num_cols: list[str] | None, exclude_cols: list[str], cat_fill: str = "unknown", num_fill: float | None = None)`
  - `SplitSection(split_date: int | dict[str, int], val_size: float = 0.2, seed: int = 42)`
- `tests/conftest.py` fixture `synthetic_df` (added in Task 2) and `config_path` → `configs/model_config.yaml`.

- [ ] **Step 1:** `git init -b main`, `git remote add origin <remote>`. Write `pyproject.toml` (hatchling, `packages=["src/train_model"]`, deps from Tech Stack, `[project.optional-dependencies] spark=["pyspark>=3.3"]`, `s3=["s3fs","boto3"]`, `[dependency-groups] dev=["pytest","ruff","ipykernel","nbconvert","nbformat"]`, pytest `testpaths=["tests"]`). `.gitignore`: `.venv/ __pycache__/ artifacts/ catboost_info/ *.cbm data/ .ipynb_checkpoints/ .pytest_cache/ .ruff_cache/`.
- [ ] **Step 2:** Write `configs/model_config.yaml` with all spec §5.1 blocks; defaults: target `GB`, id `msisdn`, date `date`, group `client`, split_date `20250701`, hard `{missing_threshold: 0.95, quasi_constant_tol: 0.98, iv_threshold: 0.02, corr_threshold: 0.9}`, soft `{top_k: 15, min_votes: 3, criteria: [all 10 names, see Task 5]}`, catboost `{iterations: 500, depth: 4, learning_rate: 0.05, eval_metric: AUC, early_stopping_rounds: 50}`, scoring `{lower: 300, upper: 850, pct_lb: 0.01, pct_ub: 0.99, exponent: 1.3, target_label: good}`, spark `{app_name: train_model_predict, master: "local[*]", conf: {}}`, paths `{artifacts_dir: artifacts}`.
- [ ] **Step 3: Failing test** `tests/test_config.py::test_load_default_config` — `cfg = load_config(config_path)`; assert `cfg.model.target_col == "GB"`, `cfg.split.split_date == 20250701`, `cfg.soft_selection["min_votes"] == 3`. And `test_missing_section_raises` — YAML without `model` → `ValueError` naming the section.
- [ ] **Step 4:** `uv venv --python 3.11 && uv sync`; run `uv run pytest tests/test_config.py -v` → FAIL (import error).
- [ ] **Step 5:** Implement `config.py` (yaml.safe_load → dataclasses; unknown keys inside dict sections kept as-is).
- [ ] **Step 6:** `uv run pytest tests/test_config.py -v` → PASS; `uv run ruff check .` → clean.
- [ ] **Step 7:** Commit `chore: scaffold train_model package and config loader` (include spec + plan docs).

### Task 2: Data — synthetic generator, split, label summary

**Files:** Create `src/train_model/data/__init__.py`, `synthetic.py`, `split.py`, `label_analysis.py`; Test `tests/test_data.py`; Modify `tests/conftest.py`.

**Interfaces:**
- Produces:
  - `make_synthetic_dataset(n_rows: int = 6000, n_months: int = 12, start: str = "20240801", seed: int = 42) -> pd.DataFrame` with columns: `msisdn`, `date` (int yyyymmdd, 1st of month), `client` (`"A"`/`"B"`), `GB` (0/1, ~20–35 % positive); informative numeric `usage_data_mb_l1m, usage_data_mb_l3m, recharge_amt_l1m, recharge_cnt_l3m, voice_out_min_l1m, sms_out_cnt_l1m, demographic_age_l1m`; categorical `device_manu, sub_type, demographic_province_l1m`; noise `noise_1..noise_5`; traps `const_col` (single value), `dup_of_usage` (exact copy of `usage_data_mb_l1m`), `corr_of_recharge` (recharge_amt_l1m × 1.01 + tiny noise), `mostly_missing` (≥97 % NaN), `all_missing` (100 % NaN). Categorical columns contain some NaN and empty strings. Module constant `INFORMATIVE_FEATURES: tuple[str, ...]` listing the 7 informative numeric names.
  - `time_split(df, date_col: str, split_date: int | dict[str, int], val_size: float = 0.2, seed: int = 42, group_col: str | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` — test = `date >= split_date` (per-group when dict, requires `group_col`); val = random (stratified on target if given) slice of the rest.
  - `label_summary(df, label: str, date_col: str, group_col: str | None = None) -> dict[str, pd.DataFrame]` keys `overall`, `by_month`, `by_group` (absent if no group_col); columns `count, n_positive, rate`.
- conftest fixture `synthetic_df` (session scope, `make_synthetic_dataset(n_rows=3000, n_months=8)`).

- [ ] **Step 1: Failing tests** in `tests/test_data.py`:
  - `test_synthetic_shape_and_traps`: 6000 rows default; `df["const_col"].nunique() == 1`; `(df["dup_of_usage"] == df["usage_data_mb_l1m"]).all()`; `df["all_missing"].isna().all()`; `0.15 < df["GB"].mean() < 0.4`; same seed → `pd.testing.assert_frame_equal`.
  - `test_time_split_by_date`: no test row with `date < split`, no train/val row with `date >= split`; `len(train)+len(val)+len(test) == len(df)`; input df unchanged (compare to copy).
  - `test_time_split_by_group_dict`: `{"A": d1, "B": d2}` respected per client.
  - `test_label_summary`: `overall["rate"].iloc[0] == pytest.approx(df.GB.mean())`; `by_month` has one row per distinct date.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. Label generation: logistic of a linear combo of standardized informative features + one category effect, seeded RNG (`np.random.default_rng`). **Step 4:** Run → PASS. **Step 5:** Commit `feat(data): synthetic dataset, time split, label summary`.

### Task 3: Preprocessing

**Files:** Create `src/train_model/preprocessing/__init__.py`, `transforms.py` (port `M/transform.py`), `missing_outliers.py` (port `M/missing_ouliers_processing.py`), `fill.py`; Test `tests/test_preprocessing.py`.

**Interfaces:**
- Produces:
  - All classes from `M/transform.py` with the same names and constructor params (`LogTransformer, PowerTransformerWrapper, QuantileBinner, PolynomialFeatureGenerator, ScalerWrapper, LabelEncoderWrapper, OneHotEncoderWrapper, FrequencyEncoder, TargetEncoder, MeanImputer, MedianImputer, ModeImputer, ConstantImputer`). Fix: fitted state lives in trailing-underscore attributes set in `fit` (not `__init__`), so `sklearn.base.clone` works.
  - `num_missing_step(numerical, method: str, arbitrary_number=0) -> tuple[str, Transformer]`, `cat_missing_step(categorical, method: str, fill_value="missing") -> tuple[str, Transformer] | None`, `num_outlier_step(method, numerical, fold=3, tail="right") -> tuple | None`, `cat_outlier_step(dict_replace) -> tuple | None`, class `CapCategoricalOutlier` (fix typo bug in cat arbitrary imputer).
  - `fill_missing(df, cat_cols: list[str], cat_fill: str = "unknown", num_fill: float | None = None) -> pd.DataFrame`: returns copy; cat cols → str, strip, lower, `""`/NaN/"nan"/"none" → `cat_fill`; numeric cols filled with `num_fill` if not None, else left NaN (CatBoost handles NaN).
- [ ] **Step 1: Failing tests:** `test_fill_missing_normalizes_cats` (`[" Apple", None, "", "SAMSUNG"]` → `["apple","unknown","unknown","samsung"]`, input unchanged); `test_fill_missing_num_fill`; `test_transformers_clone_and_fit` (parametrize over each transformer: `clone(t).fit(X, y).transform(X)` returns same index, does not mutate X); `test_cat_missing_step_arbitrary` (returns a working CategoricalImputer).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Port + fix. **Step 4:** PASS. **Step 5:** Commit `feat(preprocessing): transformers, missing/outlier steps, fill_missing`.

### Task 4: Binning/IV and HardFeatureSelector

**Files:** Create `src/train_model/feature_selection/__init__.py`, `binning.py`, `hard.py`; Test `tests/test_hard_selection.py`.

**Interfaces:**
- Produces:
  - `fit_optbinning(df, features: list[str], target: str | pd.Series, cat_cols: list[str] = (), max_n_bins: int = 10) -> dict[str, OptimalBinning]` (failures logged via `logging`, skipped).
  - `compute_iv(df, features, target, cat_cols=(), max_n_bins=10) -> pd.DataFrame[feature, iv]` sorted desc; failed features get `iv = NaN`.
  - `WoETransformer(features: list[str] | None = None, cat_cols: list[str] = (), max_n_bins: int = 10)` sklearn transformer: `fit(X, y)` stores `binnings_`; `transform(X)` returns copy with WoE values (`metric_missing="empirical"`).
  - `HardFeatureSelector(missing_threshold=0.95, quasi_constant_tol=0.98, iv_threshold=0.02, corr_threshold=0.9, cat_cols: list[str] = (), max_n_bins=10)`; `fit(X, y)` sets `selected_features_: list[str]`, `iv_: pd.Series`, `report_: pd.DataFrame[feature, step, value]` with step ∈ {`missing`, `constant`, `duplicate`, `low_iv`, `high_corr`}; `transform(X)` → `X[selected_features_]` (raw values). Order: missing → constant/quasi-constant (feature_engine `DropConstantFeatures(tol, missing_values="include")`) → duplicate (`DropDuplicateFeatures`) → IV (drop `< iv_threshold` or NaN) → corr on numeric survivors: pairs with |corr| > threshold processed by |corr| descending, drop the lower-IV member, skip pairs already touched.
- [ ] **Step 1: Failing tests** (use `synthetic_df` train part, features = all except id/date/client/GB):
  - `test_hard_selector_removes_traps`: `report_` has `all_missing`,`mostly_missing` at step `missing`; `const_col` at `constant`; `dup_of_usage` at `duplicate`; exactly one of `recharge_amt_l1m`/`corr_of_recharge` at `high_corr`; all 7 `INFORMATIVE_FEATURES` minus at most the corr loser in `selected_features_`.
  - `test_hard_selector_does_not_mutate_input`.
  - `test_hard_selector_handles_cat_with_nan` (cat cols present with NaN, fit does not raise, cats can be selected).
  - `test_hard_selector_drops_all_nan_column` (single all-NaN column df + one good col → dropped with reason, no exception).
  - `test_two_selectors_independent`: fitting a second selector on a subset leaves the first's `iv_` unchanged (no globals).
  - `test_compute_iv_informative_higher_than_noise`: mean IV of informative > mean IV of `noise_*`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (port `M/hard_feature_selection.py` + `A/models/train.py` `calc_iv`/`high_correlation_filter`/`opt_binning_*`). **Step 4:** PASS. **Step 5:** Commit `feat(selection): optbinning IV/WoE and HardFeatureSelector`.

### Task 5: SoftFeatureSelector, forward Gini, importance selection

**Files:** Create `src/train_model/feature_selection/soft.py`, `importance.py`; Test `tests/test_soft_selection.py`.

**Interfaces:**
- Consumes: none beyond sklearn/catboost.
- Produces:
  - `ALL_CRITERIA = ("correlation", "permutation", "single_feature", "tree_importance", "tree_vs_random", "variance", "mutual_info", "l1", "l2", "rfe")`.
  - `SoftFeatureSelector(top_k: int = 15, min_votes: int = 3, criteria: Sequence[str] = ALL_CRITERIA, cat_cols: Sequence[str] = (), variance_threshold: float = 0.1, mi_percentile: int = 60, n_estimators: int = 200, random_state: int = 42)`; `fit(X_train, y_train, X_val, y_val) -> self`; attributes `votes_: pd.DataFrame` (index feature, one bool column per active criterion, `n_votes` int; sorted by `n_votes` desc then name), `selected_: list[str]` (`n_votes >= min_votes`, in that order); `transform(X) -> X[selected_]`. Unknown criterion name → `ValueError`.
  - Rules per criterion (rank = `scipy.stats.rankdata(-score, method="min")`, vote if rank ≤ top_k): correlation = |Pearson r| with y; permutation = AUC drop on val (RandomForest, depth 6); single_feature = val AUC of one-feature RF (depth 3); tree_importance = RF `feature_importances_`; tree_vs_random = importance > importance of an internal random column (added to an internal copy only); variance = `VarianceThreshold(variance_threshold)` on MinMax-scaled data; mutual_info = `SelectPercentile(mutual_info_classif, mi_percentile)`; l1/l2 = `SelectFromModel(LogisticRegression(penalty, solver="liblinear"), threshold="0.1*mean")` on StandardScaled data; rfe = `RFE(DecisionTreeClassifier(max_depth=6), n_features_to_select=top_k)` rank == 1. Internal numeric matrix: cat cols ordinal-encoded, NaN filled with median (num) / -1 (cat). Input never mutated.
  - `forward_selection_gini(X_train, y_train, X_val, y_val, ordered_features: list[str], cat_cols=(), step: int = 1, max_features: int | None = None, catboost_params: dict | None = None) -> pd.DataFrame[n_features, last_feature, gini_val]`; `best_k(result: pd.DataFrame, tolerance: float = 0.005) -> int` = smallest `n_features` with `gini_val >= max - tolerance`.
  - `select_by_importance(X, y, estimator, top_k=None, threshold=None, accumulate=None, fit_kwargs: dict | None = None) -> tuple[list[str], pd.DataFrame[feature, importance, cumulative]]` — importance normalized to sum 1; priority accumulate > top_k > threshold; none given → `ValueError`. `ImportanceSelector(estimator=None, top_k=None, threshold=None, accumulate=None)` transformer (default estimator `CatBoostClassifier(iterations=300, verbose=False)`).
- [ ] **Step 1: Failing tests** (synthetic train/val, numeric + cat, small `n_estimators=50` for speed):
  - `test_soft_selector_votes_shape`: `set(votes_.columns) == set(ALL_CRITERIA) | {"n_votes"}`; `n_votes == votes_[list(ALL_CRITERIA)].sum(axis=1)`.
  - `test_soft_selector_prefers_informative`: mean `n_votes` of informative > mean of `noise_*`.
  - `test_min_votes_filters`: every feature in `selected_` has `n_votes >= min_votes`; `min_votes=len(criteria)+1` → empty list.
  - `test_soft_selector_does_not_mutate_input` (no `RANDOM_COLUMN` appears in X_train).
  - `test_subset_criteria_and_unknown_raises`.
  - `test_forward_selection_and_best_k`: on 5 ordered features returns 5 rows; `best_k` on a hand-made frame `gini=[.30,.40,.452,.455,.454]`, tolerance .005 → `3`.
  - `test_select_by_importance_accumulate_normalized`: with CatBoost and with RandomForest, `accumulate=0.9` returns a non-empty list whose cumulative ≤ 0.9 except possibly the boundary feature, and `importance.sum() == approx(1)`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (port `M/soft_feature_selection.py`, `M/group_feature_selection.py`, `A/models/train.py::select_important_features`). **Step 4:** PASS. **Step 5:** Commit `feat(selection): soft voting selector, forward gini, importance selector`.

### Task 6: Evaluation

**Files:** Create `src/train_model/evaluation/__init__.py`, `metrics.py`, `psi.py`, `shap_utils.py`, `plots.py`; Test `tests/test_evaluation.py`.

**Interfaces:**
- Produces:
  - `auc_gini(y_true, y_score) -> dict[str, float]` keys `auc, gini`; `auc_gini_by_group(df, y_true_col, y_score_col, group_col) -> pd.DataFrame[group, n, auc, gini]` (NaN when one class).
  - `decile_table(y_true, y_score, n_bins: int = 10, score: array | None = None) -> pd.DataFrame` columns `top, count, n_positive, acc_positive, positive_rate, acc_positive_rate, cover_positive, proba_range` (+ `score_range` if score given); top 1 = highest y_score; ties handled by `qcut(duplicates="drop")`.
  - `pr_auc_by_bin(y_true, y_score, n_bins=10) -> pd.DataFrame[bin, pr_auc]`.
  - `psi(expected, actual, bins: int = 10, method: Literal["quantile","uniform"] = "quantile", eps: float = 1e-6) -> float` — quantile edges from expected (unique; if <2 edges → 0.0), uniform = equal-width over combined min/max; NaN dropped; empty → `nan`.
  - `psi_by_feature(df_ref, df_cur, features, bins=10) -> pd.DataFrame[feature, psi]` (numeric only; categorical uses category frequency PSI).
  - `psi_by_bucket(ref_scores, cur_scores, bucket_edges: list[float]) -> pd.DataFrame[bucket, ref_pct, cur_pct, psi]`.
  - `shap_importance(model, X, max_samples: int = 5000) -> pd.DataFrame[feature, mean_abs_shap]`; `shap_group_importance(shap_df, sep="_") -> pd.DataFrame[group, mean, std, count, max, min]`.
  - plots (return `matplotlib.figure.Figure`, never `plt.show()`): `plot_roc(datasets: dict[str, tuple[y_true, y_score]])`, `plot_roc_by_group(df, y_true_col, y_score_col, group_col)`, `plot_shap_summary(model, X, max_display=20)`, `plot_forward_gini(result_df)`.
- [ ] **Step 1: Failing tests:** `test_auc_gini_perfect` (`gini == 1`); `test_auc_gini_by_group_single_class_nan`; `test_decile_table_monotonic` (10 rows, `count.sum()==n`, `acc_positive` last == total positives); `test_decile_table_with_ties` (y_score with 3 distinct values → ≤3 rows, no error); `test_psi_identical_zero_shifted_positive` (same sample → `< 1e-9`; shift by +1 std → `> 0.1`); `test_psi_constant_and_empty` (constant arrays → finite; empty → `isnan`); `test_psi_by_feature_handles_categorical`; `test_shap_importance_columns` (small CatBoost on synthetic; rows == n features; non-negative); `test_plots_return_figure` (use `matplotlib.use("Agg")`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (port `M/evaluation.py`, `A/models/train.py` decile/roc/shap, `A/utils/common.py::psi`, `A/models/monitor_base.py` bucket PSI). **Step 4:** PASS. **Step 5:** Commit `feat(evaluation): metrics, PSI, SHAP, plots`.

### Task 7: Modeling, bundle, CatBoost utils, pandas scoring

**Files:** Create `src/train_model/modeling/__init__.py`, `train.py`, `bundle.py`, `catboost_utils.py`, `src/train_model/scoring/__init__.py`, `score.py`; Test `tests/test_modeling.py`, `tests/test_scoring.py`.

**Interfaces:**
- Consumes: `fill_missing` (Task 3), `auc_gini` (Task 6).
- Produces:
  - `train_catboost(train: pd.DataFrame, val: pd.DataFrame, features: list[str], target: str, cat_features: list[str] = (), params: dict | None = None, seed: int = 42) -> CatBoostClassifier` (cat features cast to str; eval_set = val; `use_best_model=True`).
  - `evaluate_splits(model, splits: dict[str, pd.DataFrame], features, target) -> pd.DataFrame[split, n, auc, gini]` (does not add columns to inputs).
  - `ModelBundle` dataclass: `model: CatBoostClassifier, features: list[str], cat_features: list[str], target: str, model_code: str, cat_fill: str, num_fill: float | None, train_period: tuple[int, int] | None = None, score_bounds: tuple[float, float] | None = None, metrics: dict = {}, extra: dict = {}`; `save(dir: str | Path) -> Path` writes `model.cbm` + `meta.json` (incl. `created_at`, `train_model` version, catboost version); `ModelBundle.load(dir) -> ModelBundle`; `predict_proba(df) -> np.ndarray` (applies `fill_missing` with bundle fills, selects `features` in order).
  - `CatBoostEnsembleClassifier`, `CatBoostTransferLearner`, `CatBoostShrinker` — same API as `M/models.py`; fix missing `import copy`; TransferLearner `save`/`load` use json instead of pickle.
  - `proba_to_score(proba, target_label: Literal["good","bad"] = "good", lower=300, upper=850, pct_lb=0.01, pct_ub=0.99, exponent=1.3, bounds: tuple[float, float] | None = None) -> np.ndarray[int]` and `fit_score_bounds(proba, target_label="good", pct_lb=0.01, pct_ub=0.99, exponent=1.3) -> tuple[float, float]` (returns raw-score min/max after clip + transform, used as `bounds`). Algorithm (from `A/models/predict_base.py::transform_to_com_score`): `p = |label_int - proba|` with label_int=1 for good, 0 for bad; clip p to its [pct_lb, pct_ub] quantiles; `s = -log(p/(1-p))`; `s = sign(s)*|s|**exponent`; scale `(s - lo)/(hi - lo)*(upper-lower)+lower` with `(lo,hi)=bounds or (s.min(), s.max())`, then floor and clip to [lower, upper].
- [ ] **Step 1: Failing tests:**
  - `test_train_catboost_learns`: on synthetic informative features, test gini > 0.3.
  - `test_train_with_cat_nan`: cat cols containing NaN train without error.
  - `test_evaluate_splits_no_mutation`.
  - `test_bundle_roundtrip_predictions_equal` (save to `tmp_path`, load, `np.allclose` predictions, features/cat_features/fills equal, `meta.json` is valid JSON).
  - `test_ensemble_and_shrinker` (2 tiny models; ensemble proba shape `(n,2)`, rows sum to 1; shrinker `best_ntree_end <= tree_count_`).
  - `test_score_range_and_monotonic` (`good`: higher proba → higher-or-equal score; all in [300, 850]); `test_score_bad_label_inverts`; `test_score_fixed_bounds_stable` (scoring a subset with `bounds` from full set gives same scores as those rows in full-set scoring).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat(modeling): catboost training, model bundle, scoring`.

### Task 8: Spark subpackage, predict script, notebook 02 (not executed)

**Files:** Create `src/train_model/spark/__init__.py` (empty docstring only, no imports), `session.py`, `predict.py`, `binning.py`, `scoring.py`, `shap.py`, `scripts/predict_spark.py`, `notebooks/02_predict_spark.ipynb`; Test `tests/test_spark.py`.

**Interfaces:**
- Consumes: `ModelBundle` (Task 7), `load_config` (Task 1), `fit_optbinning` result objects (Task 4).
- Produces (port from `A/models/predict_base.py`, `A/utils/common.py`):
  - `create_spark_session(app_name: str, master: str | None = "local[*]", conf: dict[str, str] | None = None, log_level: str = "ERROR") -> SparkSession`; `s3a_conf_from_env(endpoint_var="S3_ENDPOINT", key_var="S3_ACCESS_KEY", secret_var="S3_SECRET_KEY") -> dict[str, str]` (standard fs.s3a.* keys, path-style, no hard-coded values).
  - `fill_missing_spark(df, cat_cols, cat_fill="unknown", num_fill=None) -> DataFrame` (same semantics as pandas `fill_missing`).
  - `predict_spark(spark, bundle: ModelBundle, df, id_col: str, keep_cols: Sequence[str] = (), return_features: bool = False, proba_col: str = "predict_proba") -> DataFrame` — broadcast model, `pandas_udf` returning double, features in `bundle.features` order, cat cast to string.
  - `woe_case_when_expr(column: str, binning_table: pd.DataFrame, handle_null: bool = False) -> str`; `woe_transform_spark(df, binnings: dict[str, OptimalBinning]) -> DataFrame`.
  - `proba_to_score_spark(df, proba_col="predict_proba", score_col="score", target_label="good", lower=300, upper=850, pct_lb=0.01, pct_ub=0.99, exponent=1.3, bounds=None, relative_error=1e-6) -> DataFrame` — identical formula to Task 7 `proba_to_score`.
  - `top_features_shap_spark(spark, bundle, df, top_n: int = 10, sample_size: int = 50000) -> DataFrame[model_code, feature_name, rank, shap_value]`.
  - `scripts/predict_spark.py`: argparse `--config --model-dir --input --output [--date] [--id-col]`; reads parquet, `fill_missing_spark` → `predict_spark` → `proba_to_score_spark(bounds=bundle.score_bounds)` → writes `{output}/predict[/date=..]` and `{output}/shap_value[/date=..]`, mode overwrite. Spark conf = config `spark.conf` merged with `s3a_conf_from_env()` when env vars present.
  - Notebook 02: markdown (Vietnamese) + cells: config → session → read input parquet (path from config, placeholder) → load bundle → predict → score → SHAP → write. No outputs saved.
- [ ] **Step 1:** Write `tests/test_spark.py` starting with `pyspark = pytest.importorskip("pyspark")`; tests: `test_proba_to_score_spark_matches_pandas` (local session, 200 rows, scores equal to `proba_to_score` with same bounds), `test_predict_spark_matches_bundle` (predictions ≈ `bundle.predict_proba`), `test_woe_expr_matches_optbinning` (≤1e-6 diff on numeric feature).
- [ ] **Step 2:** Implement modules, script, notebook.
- [ ] **Step 3:** Verify locally without Spark: `uv run pytest -q` → spark tests reported **skipped**, rest pass; `uv run python -c "import train_model, train_model.modeling, train_model.feature_selection"` works with pyspark absent; `uv run ruff check .` clean; `uv run python -m py_compile scripts/predict_spark.py src/train_model/spark/*.py`.
- [ ] **Step 4:** Commit `feat(spark): spark predict, scoring, woe, shap and predict script (not run locally)`.

### Task 9: Training notebook, README

**Files:** Create `notebooks/01_train.ipynb`, rewrite `README.md`.

**Interfaces:** Consumes everything from Tasks 1–7.

- [ ] **Step 1:** Build `01_train.ipynb` (generate via a small nbformat script kept in scratchpad, not in repo). Sections (Vietnamese markdown headers): 0 Setup (`load_config("../configs/model_config.yaml")`, `matplotlib inline`) · 1 Data (`make_synthetic_dataset`; comment showing how to swap in real parquet) · 2 Label summary · 3 Split (`time_split`) · 4 Fill missing · 5 Hard selection (print `report_` grouped by step, `iv_` top 20) · 6 Soft selection (`votes_` table, `selected_`) · 7 Forward Gini (`plot_forward_gini`, `best_k`) · 8 Train CatBoost on final features · 9 Evaluate (`evaluate_splits`, `plot_roc`, `decile_table` on test, `auc_gini_by_group` by client and month, `psi` train vs test score, `psi_by_feature`) · 10 SHAP (`shap_importance`, `shap_group_importance`, `plot_shap_summary`) · 11 Score (`fit_score_bounds` on train, `proba_to_score` on test, decile with score) · 12 Save bundle to `artifacts/{model_code}` and reload check · 13 Next step: link to notebook 02 / script.
- [ ] **Step 2:** Execute: `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_train.ipynb --ExecutePreprocessor.timeout=900`. Expected: exits 0; test gini printed > 0.3; `artifacts/<model_code>/model.cbm` exists.
- [ ] **Step 3:** README (Vietnamese): purpose, structure tree, setup (`uv venv --python 3.11`, `uv sync`, optional `uv sync --extra spark`), how to start a new model (clone → copy config → replace data cell → run notebook), module map with the main function per module, hard vs soft selection explained (10 criteria table + `min_votes`), Spark section (run on cluster: `spark-submit scripts/predict_spark.py ...`; not tested locally), dev commands (`uv run pytest`, `uv run ruff check .`).
- [ ] **Step 4:** Commit `docs: training notebook with outputs and README`.

### Task 10: Final verification and push

- [ ] **Step 1:** `uv run ruff check .` → `All checks passed!`; `uv run pytest -q` → all pass, spark tests skipped.
- [ ] **Step 2:** `grep -rniE "unitel|s3a://|registry\." src/ scripts/ configs/` → no matches.
- [ ] **Step 3:** `git status` clean; `git log --oneline` shows task commits.
- [ ] **Step 4:** `git push -u origin main`. If authentication fails, report to the user with the exact error and stop (do not change remote or credentials).
