"""Configuration management for chest X-ray classification project."""

import yaml


def load_config(config_path: str = "config/config.yml") -> dict:
    """Load configuration parameters from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
