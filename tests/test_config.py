from pathlib import Path

import pytest

from train_model.config import load_config


def test_load_default_config(config_path):
    cfg = load_config(config_path)
    assert cfg.model.target_col == "GB"
    assert cfg.split.split_date == 20250501
    assert cfg.soft_selection["min_votes"] == 3
    assert "device_manu" in cfg.features.cat_cols
    assert cfg.features.cat_fill == "unknown"
    assert cfg.scoring["lower"] == 300


def test_missing_section_raises(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("features:\n  cat_cols: []\nsplit:\n  split_date: 20250101\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_config(p)
