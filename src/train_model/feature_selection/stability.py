"""Temporal predictive consistency: is a feature's separation of the label stable over time?

Idea adapted (re-implemented, not copied) from the *FeConsistency* / *WOE Stabilizer* steps of
AIDflow (https://github.com/aigensys-labs/aidflow).

For every bin ``b`` of a feature and every period ``t`` (e.g. snapshot month) compute the
*target ratio*::

    TR[b, t] = rate(label | bin b, period t) / rate(label | period t) - 1

A useful feature keeps each bin on the **same side** of the period average month after month
(``TR > 0`` every month, or ``TR < 0`` every month). The consistency score is the weighted share of
(bin, period) cells that agree with the bin's dominant side:

* ``1.0`` — every bin always stays above (or always below) the period average;
* ``0.0`` — bins flip sides at random (random walk around the average → likely overfit).

Only cells that carry evidence vote:

1. a bin is kept if it separates the label at all — under the null "bin rate = period rate in every
   period", ``sum_t z_t^2`` follows a chi-square with one degree of freedom per period, where
   ``z_t = TR / se_t`` and ``se_t`` is the binomial standard error of the period rate for the cell size;
2. inside kept bins, only cells with ``|z_t| > z_(1-alpha/2)`` vote — a cell whose TR is within
   sampling noise says nothing about the direction.

Cell weights favour (a) cells with enough observations (``W1 = min(n / mean_n_of_bin, 1)``) and
(b) bins with strong leverage (``W2 = mean |TR| of the bin``). The product is normalised and clipped
at 2.7 so that no single cell dominates. A feature with no voting cell is ``no_signal``.

Differences from AIDflow (deliberate): AIDflow drops bins whose *mean* TR is not significantly
different from 0, then lets every remaining cell vote with weight ``|mean TR|``. A bin that is
strongly above average one month and strongly below the next has mean TR ~ 0, so it is dropped or
weighted ~0 — exactly the unstable behaviour the test should catch. And every cell of a kept bin
votes, so a weak-but-stable feature (small TR, cells of a few dozen rows) loses votes to cells that
flip sign by sampling noise. The two significance steps above handle both cases.

Unlike PSI (which compares *distributions*), this tests the *relationship with the label*, so it
catches features whose distribution is stable but whose meaning drifts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.base import BaseEstimator, TransformerMixin

MISSING_BIN = "__missing__"

STABLE = "stable"
UNSTABLE = "unstable"
NO_SIGNAL = "no_signal"  # no bin separates the label significantly
INSUFFICIENT = "insufficient_time"  # fewer than 2 usable periods


@dataclass
class ConsistencyResult:
    score: float
    status: str
    n_bins: int
    n_periods: int
    table: pd.DataFrame  # one row per (bin, period): count, rate, period_rate, tr, weight


# --------------------------------------------------------------------------- binning
def fit_bin_edges(x: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Quantile edges of a numeric series (outer edges open, duplicates removed)."""
    v = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(v) == 0:
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def apply_bins(x: pd.Series, edges: np.ndarray | None) -> pd.Series:
    """Map values to string bin labels. ``edges=None`` = categorical (value itself is the bin).

    Missing values always get their own bin :data:`MISSING_BIN`.
    """
    if edges is None:
        obj = x.astype(object)
        out = obj.where(obj.isna(), obj.map(str))
        return out.fillna(MISSING_BIN).astype(str)
    v = pd.to_numeric(x, errors="coerce")
    codes = pd.cut(v, edges, labels=False, include_lowest=True)
    return codes.map(lambda c: MISSING_BIN if pd.isna(c) else f"b{int(c):02d}").astype(str)


# --------------------------------------------------------------------------- core test
def temporal_consistency(
    bins: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    period: pd.Series | np.ndarray,
    *,
    min_bin_obs: int = 20,
    alpha: float = 0.05,
    threshold: float = 0.65,
) -> ConsistencyResult:
    """Consistency score of an already-binned feature (see module docstring).

    ``bins`` holds one bin label per row, ``y`` the 0/1 label, ``period`` the time unit (e.g. month).
    Cells with ``count <= min_bin_obs`` are ignored. ``alpha`` is the significance level of both the
    per-bin chi-square test and the per-cell z-test.
    """
    df = pd.DataFrame({"bin": np.asarray(bins), "y": np.asarray(y, dtype=float), "t": np.asarray(period)})
    cells = df.groupby(["bin", "t"], observed=True)["y"].agg(rate="mean", count="size").reset_index()
    period_rate = df.groupby("t", observed=True)["y"].mean().rename("period_rate")
    cells = cells.merge(period_rate, left_on="t", right_index=True)
    cells = cells[(cells["count"] > min_bin_obs) & (cells["period_rate"] > 0)].copy()
    cells["tr"] = cells["rate"] / cells["period_rate"] - 1

    n_periods = int(cells["t"].nunique())
    if n_periods < 2:
        return ConsistencyResult(np.nan, INSUFFICIENT, int(cells["bin"].nunique()), n_periods, cells)

    # denoise, step 1: keep bins whose rate differs from the period rate at all (chi-square over periods)
    pr = cells["period_rate"].clip(upper=1 - 1e-9)
    se = np.sqrt(pr * (1 - pr) / cells["count"]) / pr
    cells["z"] = cells["tr"] / se
    stats = cells.assign(z2=cells["z"] ** 2).groupby("bin").agg(chi2=("z2", "sum"), n_t=("t", "size"))
    stats["p_value"] = chi2.sf(stats["chi2"], stats["n_t"])
    signal_bins = stats.index[stats["p_value"] < alpha]
    # step 2: only cells significantly away from the period average vote (sampling noise does not)
    cells = cells[cells["bin"].isin(signal_bins) & (cells["z"].abs() > norm.ppf(1 - alpha / 2))]
    cells = cells.drop(columns="z")
    if cells.empty:
        return ConsistencyResult(np.nan, NO_SIGNAL, 0, n_periods, cells)

    # weights
    mean_count = cells.groupby("bin")["count"].transform("mean")
    w1 = (cells["count"] / mean_count).clip(upper=1.0)
    w2 = cells.groupby("bin")["tr"].transform(lambda s: s.abs().mean())
    w = w1 * w2
    w = (w / w.mean()).clip(upper=2.7) if w.mean() > 0 else w
    cells = cells.assign(weight=w)

    above = cells["weight"].where(cells["tr"] > 0, 0.0).groupby(cells["bin"]).sum()
    below = cells["weight"].where(cells["tr"] < 0, 0.0).groupby(cells["bin"]).sum()
    total = cells["weight"].sum()
    score = float((above - below).abs().sum() / total) if total > 0 else 0.0
    status = STABLE if score >= threshold else UNSTABLE
    return ConsistencyResult(score, status, int(cells["bin"].nunique()), n_periods, cells.reset_index(drop=True))


# --------------------------------------------------------------------------- selector
class TemporalStabilitySelector(BaseEstimator, TransformerMixin):
    """Drop features whose relation with the label is not consistent over time.

    ``X`` must contain ``time_col`` (e.g. the snapshot ``date``); every other column is tested.
    Numeric features are cut into ``n_bins`` quantile bins learnt on the fit sample; categorical
    features use their categories. Missing values are their own bin.

    ``fit(X, y, X_val=None, y_val=None)``: when a second sample is given (e.g. the validation split),
    the same bins are applied to it and a feature must be consistent on **both** samples — this guards
    against a feature looking consistent by chance on one sample.

    Do **not** pass the final out-of-time test set: choosing features with the test labels makes the
    test metrics optimistic (leakage). The test set must only be used to evaluate the result.

    Result attributes: ``report_`` (score / status per sample, ``keep``), ``selected_features_``,
    ``tables_`` (per-feature (bin, period) table on the fit sample, for plots).

    Status ``no_signal`` / ``insufficient_time`` cannot be judged; they are kept unless
    ``drop_no_signal=True``.
    """

    def __init__(
        self,
        time_col: str,
        threshold: float = 0.65,
        n_bins: int = 10,
        min_bin_obs: int = 20,
        alpha: float = 0.05,
        cat_cols: Sequence[str] = (),
        drop_no_signal: bool = False,
    ):
        self.time_col = time_col
        self.threshold = threshold
        self.n_bins = n_bins
        self.min_bin_obs = min_bin_obs
        self.alpha = alpha
        self.cat_cols = cat_cols
        self.drop_no_signal = drop_no_signal

    def _is_cat(self, X: pd.DataFrame, col: str) -> bool:
        return col in self.cat_cols or not pd.api.types.is_numeric_dtype(X[col])

    def _test(self, X: pd.DataFrame, y, feat: str) -> ConsistencyResult:
        return temporal_consistency(
            apply_bins(X[feat], self.edges_[feat]),
            y,
            X[self.time_col],
            min_bin_obs=self.min_bin_obs,
            alpha=self.alpha,
            threshold=self.threshold,
        )

    def _keep(self, status: str | None) -> bool:
        if status is None or status == STABLE:
            return True
        if status == UNSTABLE:
            return False
        return not self.drop_no_signal

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None):
        if self.time_col not in X.columns:
            raise ValueError(f"time_col {self.time_col!r} not in X")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be given together")
        features = [c for c in X.columns if c != self.time_col]
        self.edges_ = {f: None if self._is_cat(X, f) else fit_bin_edges(X[f], self.n_bins) for f in features}

        rows, tables = [], {}
        for f in features:
            res = self._test(X, y, f)
            tables[f] = res.table
            row = {"feature": f, "score": res.score, "status": res.status, "n_bins": res.n_bins}
            val_status = None
            if X_val is not None:
                res_val = self._test(X_val, y_val, f)
                row.update(score_val=res_val.score, status_val=res_val.status)
                val_status = res_val.status
            row["keep"] = self._keep(res.status) and self._keep(val_status)
            rows.append(row)

        self.report_ = pd.DataFrame(rows).sort_values("score", na_position="last").reset_index(drop=True)
        self.tables_ = tables
        self.selected_features_ = [f for f in features if bool(self.report_.set_index("feature").loc[f, "keep"])]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)
