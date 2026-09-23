"""Load YAML configuration with the small inherited dataset blocks used here."""

from pathlib import Path

import yaml
from easydict import EasyDict


def merge_new_config(config, values):
    for key, value in values.items():
        if key == "_base_" and isinstance(value, str):
            base = yaml.safe_load(Path(value).read_text(encoding="utf-8"))
            config[key] = merge_new_config(EasyDict(), base)
        elif isinstance(value, dict):
            target = config.get(key, EasyDict())
            config[key] = merge_new_config(target, value)
        else:
            config[key] = value
    return config


def cfg_from_yaml_file(path):
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return merge_new_config(EasyDict(), values)
