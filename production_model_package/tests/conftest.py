import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    try:
        return load_dataset(file_name=config.app_config.test_data_file)
    except FileNotFoundError:  # pragma: no cover - setup guidance
        pytest.skip(
            f"dataset '{config.app_config.test_data_file}' is missing. It is "
            "fetched from Kaggle, not committed: run `tox -e fetch_data` first."
        )
