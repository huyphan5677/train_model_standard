from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def config_path() -> Path:
    return ROOT / "configs" / "model_config.yaml"
