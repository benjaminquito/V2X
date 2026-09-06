from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "training"):
        if section not in config:
            raise ConfigError(f"Missing required configuration section: {section}")

    data = config["data"]
    model = config["model"]
    split = data.get("split", {})
    ratios = [float(split.get(name, -1)) for name in ("train", "validation", "test")]
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ConfigError("data.split train/validation/test values must be positive and sum to 1")
    if int(data.get("sequence_length", 0)) < 2:
        raise ConfigError("data.sequence_length must be at least 2")
    if float(data.get("graph_radius_m", 0)) <= 0:
        raise ConfigError("data.graph_radius_m must be positive")
    if len(data.get("feature_columns", [])) != int(model.get("input_dim", -1)):
        raise ConfigError("model.input_dim must match the number of data.feature_columns")
    if int(model.get("num_classes", 0)) != 3:
        raise ConfigError("This implementation requires exactly three risk classes")


def with_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with recursively merged overrides."""

    merged = deepcopy(config)

    def merge(target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value

    merge(merged, overrides)
    validate_config(merged)
    return merged
