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

    # The model drops rows that are missing a required feature, so the
    # response can be shorter than the request. It must never be longer,
    # and it must not be empty.
    # NOTE: the client currently cannot tell WHICH rows were dropped -
    # see the TODO in regression_model/processing/validation.py. Once that
    # is reported, tighten this into an exact per-row assertion.
    assert 0 < len(prediction_data['predictions']) <= len(payload['inputs'])

    # rel_tol is RELATIVE: rel_tol=100 allowed a 100x deviation, which made
    # this assertion pass for any number at all (even a negative price).
    # 5% around the expected value is a real regression guard.
    assert math.isclose(
        prediction_data['predictions'][0], 113422, rel_tol=0.05
    )

    # Sanity: a house price model must never emit non-positive values.
    assert all(p > 0 for p in prediction_data['predictions'])