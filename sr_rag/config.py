import yaml
from pathlib import Path

class ConfigNode:
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        res = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                res[key] = value.to_dict()
            else:
                res[key] = value
        return res

def load_config(config_path: str = "config.yaml") -> ConfigNode:
    """Load configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ConfigNode(data)
