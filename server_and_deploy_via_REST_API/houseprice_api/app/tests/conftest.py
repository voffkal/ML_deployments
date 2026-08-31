from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset

from app.main import app

@pytest.fixture(scope='module')
def test_data() -> pd.DataFrame:
    try:
        return load_dataset(file_name=config.app_config.test_data_file)
    except FileNotFoundError:  # pragma: no cover - setup guidance
        pytest.skip(
            f"dataset '{config.app_config.test_data_file}' is missing. "
            "It is fetched from Kaggle, not committed: run "
            "`tox -e fetch_data` in the model package first."
        )

@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}