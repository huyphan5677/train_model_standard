"""ModelBundle: a trained CatBoost model plus everything needed to reproduce its preprocessing.

Saved as a directory::

    <dir>/model.cbm    CatBoost native model
    <dir>/meta.json    features, categorical features, fill values, score bounds, metrics, versions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from train_model.modeling.train import prepare_features
from train_model.preprocessing.fill import fill_missing

MODEL_FILE = "model.cbm"
META_FILE = "meta.json"


@dataclass
class ModelBundle:
    model: CatBoostClassifier
    features: list[str]
    cat_features: list[str]
    target: str
    model_code: str
    cat_fill: str = "unknown"
    num_fill: float | None = None
    train_period: tuple[int, int] | None = None
    score_bounds: tuple[float, float] | None = None
    metrics: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the training-time missing-value handling and return the model input frame."""
        filled = fill_missing(df, self.cat_features, self.cat_fill, self.num_fill)
        return prepare_features(filled, self.features, self.cat_features)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of class 1 for each row of ``df`` (raw features; preprocessing applied here)."""
        return self.model.predict_proba(self.preprocess(df))[:, 1]

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        import catboost

        import train_model

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(directory / MODEL_FILE))
        meta = {
            "model_code": self.model_code,
            "target": self.target,
            "features": list(self.features),
            "cat_features": list(self.cat_features),
            "cat_fill": self.cat_fill,
            "num_fill": self.num_fill,
            "train_period": list(self.train_period) if self.train_period else None,
            "score_bounds": list(self.score_bounds) if self.score_bounds else None,
            "metrics": self.metrics,
            "extra": self.extra,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "versions": {"train_model": train_model.__version__, "catboost": catboost.__version__},
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float),
                                           encoding="utf-8")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        model = CatBoostClassifier()
        model.load_model(str(directory / MODEL_FILE))
        return cls(
            model=model,
            features=meta["features"],
            cat_features=meta["cat_features"],
            target=meta["target"],
            model_code=meta["model_code"],
            cat_fill=meta.get("cat_fill", "unknown"),
            num_fill=meta.get("num_fill"),
            train_period=tuple(meta["train_period"]) if meta.get("train_period") else None,
            score_bounds=tuple(meta["score_bounds"]) if meta.get("score_bounds") else None,
            metrics=meta.get("metrics", {}),
            extra=meta.get("extra", {}),
        )
