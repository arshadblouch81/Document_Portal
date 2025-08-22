import yaml
from pathlib import Path

def load_config_old(config_path: str = "config\config.yaml") -> dict:
    with open(config_path, "r") as file:
        config=yaml.safe_load(file)
    return config

def load_config(config_path: str = "config\config.yaml") -> dict:
    # Get the path relative to this script
    config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'

    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        return config