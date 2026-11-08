"""Convert predicted probabilities into a bounded integer score (e.g. 300–850).

Algorithm (same as the Spark version in ``train_model.spark.scoring``):

1. ``p = |label_int - proba|`` with ``label_int = 1`` for ``target_label="good"`` (label 1 = good customer)
   and ``0`` for ``"bad"`` — so ``p`` is always the probability of being *bad*.
2. Without ``bounds``: clip ``p`` to its own [``pct_lb``, ``pct_ub``] quantiles.
3. ``s = -logit(p) = log((1-p)/p)`` (higher = better), then signed power ``sign(s)*|s|**exponent``.
4. Linearly map ``[lo, hi]`` to ``[lower, upper]``, floor, clip.
   ``(lo, hi)`` = ``bounds`` if given (fit once on training data with :func:`fit_score_bounds` so the
   scale is stable across months), else the min/max of the current batch (legacy behaviour).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

_EPS = 1e-12


def _label_int(target_label: str) -> int:
    if target_label == "good":
        return 1
    if target_label == "bad":
        return 0
    raise ValueError(f"target_label must be 'good' or 'bad', got {target_label!r}")


def _raw_score(proba, target_label: str, pct_lb: float, pct_ub: float, exponent: float, clip_quantiles: bool):
    p = np.abs(_label_int(target_label) - np.asarray(proba, dtype=float))
    if clip_quantiles:
        lo, hi = np.quantile(p, [pct_lb, pct_ub])
        p = np.clip(p, lo, hi)
    p = np.clip(p, _EPS, 1 - _EPS)
    s = -np.log(p / (1 - p))
    return np.sign(s) * np.abs(s) ** exponent


def fit_score_bounds(
    proba,
    target_label: Literal["good", "bad"] = "good",
    pct_lb: float = 0.01,
    pct_ub: float = 0.99,
    exponent: float = 1.3,
) -> tuple[float, float]:
    """Raw-score (lo, hi) on reference (training) predictions — store in ``ModelBundle.score_bounds``."""
    s = _raw_score(proba, target_label, pct_lb, pct_ub, exponent, clip_quantiles=True)
    return float(s.min()), float(s.max())


def proba_to_score(
    proba,
    target_label: Literal["good", "bad"] = "good",
    lower: int = 300,
    upper: int = 850,
    pct_lb: float = 0.01,
    pct_ub: float = 0.99,
    exponent: float = 1.3,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Integer scores in ``[lower, upper]``; higher score = lower probability of being bad."""
    s = _raw_score(proba, target_label, pct_lb, pct_ub, exponent, clip_quantiles=bounds is None)
    lo, hi = bounds if bounds is not None else (s.min(), s.max())
    if hi <= lo:
        return np.full(len(s), (lower + upper) // 2, dtype=int)
    # floor(x + 0.5) == Spark's F.floor(F.round(x)) (round half up) for the non-negative part of the range
    scaled = np.floor((s - lo) * (upper - lower) / (hi - lo) + 0.5) + lower
    return np.clip(scaled, lower, upper).astype(int)
