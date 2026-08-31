from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import classification_model

PACKAGE_ROOT = Path(classification_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    package_name: str
    raw_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    target: str
    unused_fields: Sequence[str]
    features: Sequence[str]
    test_size: float
    random_state: int
    numerical_vars: Sequence[str]
    categorical_vars: Sequence[str]
    cabin_vars: Sequence[str]


class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig


def find_config_file():
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"No config file found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"No config file found at {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()
