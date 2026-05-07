import yaml
from pathlib import Path


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as yml_file:
        return yaml.safe_load(yml_file)


def detect_list_key(detect: dict) -> str | None:
    for first_key, item in detect.items():
        if isinstance(item, list) and item and isinstance(item[0], dict):
            return first_key
    return None


def detect_name_key(detect_names: dict) -> str | None:
    for key, item in detect_names.items():
        if isinstance(item, str):
            return key
    return None


def flatten_record(record: dict, skip_keys: set) -> dict:
    return {key: item for key, item in record.items() if key not in skip_keys}