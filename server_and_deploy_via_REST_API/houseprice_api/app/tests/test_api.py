import math

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    payload = {
        'inputs': test_data.replace({np.nan: None}).to_dict(orient='records')
    }

    response = client.post(
        'http://localhost:8001/api/v1/predict',
        json = payload
    )

    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data['predictions']
    assert prediction_data['errors'] is None

    # Rows missing a required feature cannot be scored, so the response can
    # be shorter than the request - but every missing row must be accounted
    # for by Id, so the caller can realign results with what it sent.
    assert 0 < len(prediction_data['predictions']) <= len(payload['inputs'])
    assert len(prediction_data['predictions']) + len(
        prediction_data['dropped_ids']
    ) == len(payload['inputs'])

    # Dropping a row is not a validation error: the shipped Kaggle test set
    # always contains a few, and the request as a whole is still valid.
    assert prediction_data['errors'] is None

    # rel_tol is RELATIVE: rel_tol=100 allowed a 100x deviation, which made
    # this assertion pass for any number at all (even a negative price).
    # 5% around the expected value is a real regression guard.
    assert math.isclose(
        prediction_data['predictions'][0], 113422, rel_tol=0.05
    )

    # Sanity: a house price model must never emit non-positive values.
    assert all(p > 0 for p in prediction_data['predictions'])