"""
General-purpose utility functions shared across pipeline phases.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load the project'"'"'s YAML config file into a Python dict.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path.resolve()}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def frame_index_to_timestamp(frame_index: int, fps: float) -> float:
    """
    Convert a frame index to a timestamp in seconds.
    """
    if fps <= 0:
        raise ValueError("fps must be greater than 0")
    return frame_index / fps


def ensure_dir(path: str) -> Path:
    """
    Create a directory (and parents) if it doesn'"'"'t already exist.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
