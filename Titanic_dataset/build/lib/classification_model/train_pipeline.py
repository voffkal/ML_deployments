import sys

sys.path.append('/Users/voffkal/Desktop/Jupiter_practice/ML_deployments/Titanic_dataset')

from sklearn.model_selection import train_test_split
from classification_model.config.core import config
from classification_model.pipeline import titanic_pipe
from classification_model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    data = load_dataset(file_name = config.app_config.raw_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size = config.model_config.test_size,
        random_state = config.model_config.random_state
    )

    titanic_pipe.fit(X_train, y_train)

    save_pipeline(pipeline_to_persist=titanic_pipe)

if __name__ == '__main__':
    run_training()