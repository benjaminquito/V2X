from pathlib import Path

import pytest

from v2x_risk.config import ConfigError, load_config, with_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_default_config_is_valid() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    assert config["data"]["sequence_length"] == 15
    assert config["data"]["graph_radius_m"] == 20.0
    assert config["model"]["num_classes"] == 3


def test_invalid_split_is_rejected() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    with pytest.raises(ConfigError):
        with_overrides(config, {"data": {"split": {"train": 0.6}}})
