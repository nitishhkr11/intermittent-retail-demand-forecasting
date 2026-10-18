"""Configuration loading.

Provides a single function, `load_config`, that reads the project's
`config.yaml` and returns its contents as a plain dictionary. Every
other module gets its paths and parameters through this function
rather than hardcoding values — see config.yaml for what's available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

# src/utils/config.py -> src/utils -> src -> project root
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(config_path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load the project configuration file.

    Args:
        config_path: Path to a YAML config file. Defaults to the
            config.yaml at the project root.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file is valid YAML but not a mapping.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{config_path}'. "
            "Make sure config.yaml exists at the project root, or pass "
            "an explicit path to load_config()."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file '{config_path}' did not parse to a dictionary "
            f"(got {type(config).__name__})."
        )
    return config
