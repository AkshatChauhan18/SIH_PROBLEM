"""
src/utils/config.py
Configuration loader for SIH26153 World Model.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
