import sys
sys.path.append('/Users/voffkal/Desktop/Jupiter_practice/ML_deployments/Titanic_dataset')

from classification_model.config.core import config

from classification_model.processing.features import ExtractLetterTransformer

def test_temporal_variable_transformer(sample_input_data):
    transformer = ExtractLetterTransformer(
        variables = config.model_config.cabin_vars
    )

    assert sample_input_data['cabin'].iat[6] == 'E12'

    subject = transformer.fit_transform(sample_input_data)

    assert subject['cabin'].iat[6] == 'E'