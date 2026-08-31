# ML Deployments — from a notebook model to a deployable service

Four progressive stages of packaging and serving ML models, built while working
through a deployment course. Each directory is a step, not a separate product.

| Stage | Directory | What it adds |
|---|---|---|
| 1 | `production_model_package` | House-price regression as an installable Python package (`setup.py`, versioned artifact, tests) |
| 2 | `server_and_deploy_via_REST_API` | FastAPI service on top of that package, deployable to a PaaS |
| 3 | `deploying_with_containers` | The same API in Docker, with the model package built alongside it |
| 4 | `Titanic_dataset` | The packaging pattern reapplied to a classification problem |

## Models

Both models are deliberately simple — the subject here is the delivery pipeline,
not model quality.

| Model | Algorithm | Held-out performance |
|---|---|---|
| House price (`regression_model`) | Lasso on log-price, `feature-engine` preprocessing | RMSE ≈ \$32.6k, MAE ≈ \$18.5k, R² ≈ 0.846 (n=146) |
| Titanic (`classification_model`) | Logistic regression (C=0.0005) | accuracy ≈ 0.718, ROC-AUC ≈ 0.804 (n=131) |

Measured on the split defined in each package's config, seeded via
`random_state`. Reproduce with `tox` (see below).

## Architecture

The model package is the unit of deployment: the API depends on it as an
ordinary versioned dependency (`vve-regression-model==0.0.2`) installed from a
private index, rather than importing source from a sibling directory.

```
model_package  ──build──>  wheel  ──publish──>  private index
                                                     │
                                              pip install
                                                     ↓
                                          houseprice_api (FastAPI)
                                                     │
                                                   Docker
```

This is what makes the API independently deployable and the model independently
versioned: the pinned dependency records exactly which model a given API build
was serving.

## Running it

Each package uses `tox` as the entry point.

```bash
# model package: trains, then tests
cd production_model_package && tox

# API tests
cd deploying_with_containers/houseprice_api && tox

# container
cd deploying_with_containers
docker build --build-arg PIP_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL" -t houseprice-api .
docker run -p 8001:8001 houseprice-api
```

`PIP_EXTRA_INDEX_URL` carries the credentials for the private package index and
must come from the environment. Datasets are fetched from Kaggle and are not
committed.

Once running: `http://localhost:8001/docs` for the OpenAPI UI,
`GET /api/v1/health` for a liveness check.

## Known limitations

Kept deliberately, since this is coursework rather than a production service:

- **Dropped rows are not reported.** `validate_inputs` removes rows missing a
  required feature, so a batch of 1459 rows returns 1449 predictions with no
  indication of which ten were dropped. See the `TODO` in
  `regression_model/processing/validation.py`.
- Model selection is not tuned — no hyperparameter search, no cross-validation
  beyond a single split.
- The Titanic package duplicates the regression package's structure instead of
  sharing a common scaffold.
