"""Load configuration."""
import yaml


def load_config(config_path="config/config.yaml") -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
