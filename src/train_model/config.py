"""Load and validate a model YAML config (see ``configs/model_config.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("model", "features", "split")


@dataclass
class ModelSection:
    model_code: str
    target_col: str
    id_col: str
    date_col: str
    group_col: str | None = None


@dataclass
class FeatureSection:
    cat_cols: list[str] = field(default_factory=list)
    num_cols: list[str] | None = None
    exclude_cols: list[str] = field(default_factory=list)
    cat_fill: str = "unknown"
    num_fill: float | None = None


@dataclass
class SplitSection:
    split_date: int | dict[str, int]
    val_size: float = 0.2
    seed: int = 42


@dataclass
class ModelConfig:
    model: ModelSection
    features: FeatureSection
    split: SplitSection
    hard_selection: dict[str, Any] = field(default_factory=dict)
    soft_selection: dict[str, Any] = field(default_factory=dict)
    catboost: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    spark: dict[str, Any] = field(default_factory=dict)
    paths: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None

    def feature_columns(self, columns: list[str]) -> list[str]:
        """Resolve the candidate feature list from the columns of a DataFrame."""
        excluded = set(self.features.exclude_cols) | {self.model.target_col}
        if self.features.num_cols is not None:
            wanted = list(self.features.num_cols) + list(self.features.cat_cols)
            return [c for c in wanted if c in columns and c not in excluded]
        return [c for c in columns if c not in excluded]


def _build(cls: type, section: str, data: Any) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping, got {type(data).__name__}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid config section '{section}': {exc}") from exc


def load_config(path: str | Path) -> ModelConfig:
    """Read a YAML config file into a :class:`ModelConfig`.

    Raises ``ValueError`` when a required section (model, features, split) is missing or malformed.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    missing = [s for s in REQUIRED_SECTIONS if s not in raw]
    if missing:
        raise ValueError(f"Config {path} is missing required section(s): {', '.join(missing)}")

    features = dict(raw["features"] or {})
    features["cat_cols"] = list(features.get("cat_cols") or [])
    features["exclude_cols"] = list(features.get("exclude_cols") or [])

    return ModelConfig(
        model=_build(ModelSection, "model", raw["model"]),
        features=_build(FeatureSection, "features", features),
        split=_build(SplitSection, "split", raw["split"]),
        hard_selection=dict(raw.get("hard_selection") or {}),
        soft_selection=dict(raw.get("soft_selection") or {}),
        catboost=dict(raw.get("catboost") or {}),
        scoring=dict(raw.get("scoring") or {}),
        spark=dict(raw.get("spark") or {}),
        paths=dict(raw.get("paths") or {}),
        source=path,
    )
