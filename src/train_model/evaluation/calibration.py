"""Calibration: does the predicted probability match the observed event rate, bin by bin?

Idea adapted from AIDflow's *CalibrationAssessment* (re-implemented): bin the predictions, then test
in each bin whether the observed rate differs significantly from the mean prediction. The share of
bins **without** a significant gap is the calibration score (1.0 = calibrated everywhere).
A Hosmer–Lemeshow chi-square test over the same bins is reported as the global test.

Each bin is tested at level ``alpha``, so even a perfectly calibrated model has on average
``alpha * n_bins`` flagged bins (0.5 with the defaults): read a single flagged bin with care and rely
on the Hosmer–Lemeshow p-value for the overall verdict.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

_EPS = 1e-9


def _bin_codes(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "quantile":
        return pd.qcut(pd.Series(p).rank(method="first"), q=n_bins, labels=False).to_numpy()
    if strategy == "uniform":
        return np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")


def calibration_table(
    y_true,
    y_proba,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-bin calibration check, bins ordered by increasing predicted probability.

    Columns: ``bin, count, proba_mean, event_rate, diff (= event_rate - proba_mean), z, p_value,
    significant, direction`` where ``direction`` is ``over`` (model predicts too high), ``under``
    (too low) or ``ok``. ``z`` uses the binomial standard error of the predicted rate.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    df = pd.DataFrame({"y": y, "p": p, "bin": _bin_codes(p, n_bins, strategy)})
    t = df.groupby("bin").agg(count=("y", "size"), proba_mean=("p", "mean"), event_rate=("y", "mean")).reset_index()
    pm = t["proba_mean"].clip(_EPS, 1 - _EPS)
    se = np.sqrt(pm * (1 - pm) / t["count"])
    t["diff"] = t["event_rate"] - t["proba_mean"]
    t["z"] = t["diff"] / se
    t["p_value"] = 2 * norm.sf(t["z"].abs())
    t["significant"] = t["p_value"] < alpha
    t["direction"] = np.where(~t["significant"], "ok", np.where(t["diff"] < 0, "over", "under"))
    return t


def calibration_summary(
    y_true,
    y_proba,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    alpha: float = 0.05,
) -> dict[str, float]:
    """``calibration_score`` (share of bins with no significant gap), ``hl_stat`` / ``hl_p_value``
    (Hosmer–Lemeshow, df = n_bins - 2), ``brier`` and ``mean_proba`` vs ``event_rate``."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    t = calibration_table(y, p, n_bins, strategy, alpha)
    pm = t["proba_mean"].clip(_EPS, 1 - _EPS)
    hl = float((((t["event_rate"] - pm) ** 2) * t["count"] / (pm * (1 - pm))).sum())
    dof = max(len(t) - 2, 1)
    return {
        "calibration_score": float((~t["significant"]).mean()),
        "hl_stat": hl,
        "hl_p_value": float(chi2.sf(hl, dof)),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_proba": float(p.mean()),
        "event_rate": float(y.mean()),
    }
