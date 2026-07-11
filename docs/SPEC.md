# Basic ML API Specification

This specification describes the current API and runtime behavior implemented by the Basic ML API service.

## Service Overview

Basic ML API is a Python 3.11 Flask application that stores uploaded TensorFlow/Keras models, loads models on demand, caches loaded models in memory, and returns prediction results as JSON.

The Flask app is created by `src.main:create_app()` and registers these surfaces:

- `POST /upload_model`
- `GET /get_model`
- `POST /predict`
- `GET /health`
- `GET /metrics`

## Runtime Configuration

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_STORE_PATH` | Directory for uploaded and extracted model files | `data/model_` |
| `LOG_LEVEL` | Application log level | `INFO` |
| `LOKI_URL` | Optional Loki logging endpoint | unset |
| `APP_NAME` | Logging/monitoring app tag | `ml-api` |
| `APP_ENV` | Logging/monitoring environment tag | `dev` |
| `MODEL_CLEANUP_INTERVAL` | Cleanup interval in hours | `5` |
| `SERVER_HOST` | Flask host binding | `0.0.0.0` |
| `SERVER_PORT` | Flask port | `5000` |
| `MAX_MODEL_FILE_SIZE` | Maximum uploaded model file size | `104857600` |

## Model Upload

### Endpoint

`POST /upload_model?hash={model_hash}`

### Request

- Content type: `multipart/form-data`
- Multipart field: `model_file`
- Query parameter: `hash`
- Uploaded file format: ZIP archive containing at least one `.keras` file

### Success Response

- Status: `200`
- Body: `{"message": "Model uploaded successfully"}`

### Error Behavior

- Missing `model_file` or `hash`: `400`, `{"error": "Missing data (file or hash)"}`
- Invalid hash: `400`, `{"error": "Invalid model hash"}`
- Invalid ZIP or unsafe zip entry: `400`, error message body
- Missing `.keras`: `400`, `{"error": "No .keras file in zip"}`
- Storage failure: `500`, error message body

### Storage Behavior

- The service creates a directory under `MODEL_STORE_PATH` named by the supplied hash.
- The uploaded ZIP is temporarily saved as `temp.zip` inside a staging directory.
- ZIP contents are first validated and extracted into a temporary staging directory.
- The target hash directory is replaced through a same-filesystem backup-and-rename transaction so stale files are removed without discarding the last working model first.
- If installing the staged directory fails, the previous directory is restored and its metadata and cache entry remain usable.
- If the process stops during replacement, startup restores an unfinished backup or removes a leftover backup after a completed install.
- `temp.zip` is removed after extraction attempt.
- If `hash` already exists, the old in-memory cached model is invalidated so the next `/predict` for that hash loads from disk again.
- If a hash is uploaded again, the existing directory contents are replaced and only the new upload is retained.
- Model metadata is stored in memory with `file_path` and `used` timestamp.
- Uploads larger than `MAX_MODEL_FILE_SIZE` are rejected with `413` and `{"error": "Uploaded file too large"}`.
- On startup, only valid hash directories containing a regular `.keras` file are registered; internal staging and incomplete directories are ignored.

## Model Lookup

### Endpoint

`GET /get_model?hash={model_hash}`

### Success Response

- Status: `200`
- Body contains the stored metadata under `message`.

### Error Behavior

- Missing `hash`: `400`, `{"error": "Model hash is required"}`
- Unknown hash: `404`, `{"error": "No such model"}`
- Unexpected lookup failure: `500`, error message body

## Prediction

### Endpoint

`POST /predict?hash={model_hash}`

### Request

- Content type: `application/json`
- Query parameter: `hash`
- Body: JSON array-like prediction input data

### Success Response

- Status: `200`
- Body: `{"prediction": [...]}`
- Prediction output is converted from NumPy/TensorFlow output to JSON lists.

### Error Behavior

- Missing `hash` or empty JSON data: `400`, `{"error": "Missing hash or data"}`
- Unknown model hash: `404`, `{"error": "Model not found"}`
- TensorFlow loading or prediction failure: `500`, `{"error": "Internal error during prediction"}`

### Model Loading Behavior

- The service checks the in-memory model cache first.
- On cache hit, the cached model is reused and moved to the most-recently-used position.
- On cache miss, the service searches the model directory recursively for a `.keras` file.
- The first matching `.keras` file is loaded with `tf.keras.models.load_model`.
- The loaded model is inserted into the cache.
- The model metadata `used` timestamp is updated after prediction begins successfully.

## Model Cache

- Cache type: in-memory LRU cache backed by `OrderedDict`.
- Default maximum size: 10 models.
- When the cache is full, the least-recently-used model is evicted before loading a new model.
- Cache state is process-local and is not shared across multiple workers.

## Stale Model Cleanup

- A background daemon thread runs cleanup on each `ModelManager` instance.
- Cleanup removes models whose `used` timestamp is older than one week.
- Cleanup deletes the model directory from disk, removes metadata, and removes cached model instances.
- Cleanup interval is configurable through `MODEL_CLEANUP_INTERVAL`.

## Health Endpoint

### Endpoint

`GET /health`

### Success Response

- Status: `200`
- Body: `Healthy`

## Metrics Endpoint

### Endpoint

`GET /metrics`

### Response

- Status: `200`
- Content type: Prometheus text format

### Expected Metrics

- `model_cache_usage`
- `predictions_completed`
- `cache_hits`
- `cache_misses`
- `ml_api_errors_total`
- `ml_api_cpu_usage_percent`
- `ml_api_ram_usage_mb`

## Deployment Notes

- The Docker image runs Gunicorn with one worker: `gunicorn -w 1 -b 0.0.0.0:5000 src.main:create_app()`.
- Docker Compose maps host port `12021` to container port `5000`.
- Docker Compose persists model data through `./data:/usr/src/app/data`.
- Docker Compose defaults image tags to `dev` when `VERSION` is unset and still accepts a local `.env` override.
- Logging and resource monitoring are initialized during app creation.

## Current Test Coverage

The test suite currently covers:

- Health endpoint response.
- Metrics endpoint content and metric names.
- Successful model upload with a ZIP containing `.keras`.
- Upload validation for missing data.
- Successful prediction with mocked TensorFlow loading.
- Prediction validation for missing, empty, malformed, and unknown-model requests.
- Model lookup success, missing hash, and unknown model behavior.

Run tests from the repository root with:

```bash
uv run python -m pytest
```

## Known Implementation Constraints

- The service has no built-in authentication or authorization.
- Metadata is in memory and rebuilt from model directories at startup.
- ZIP extraction should be reviewed before accepting untrusted uploads.
- Uploaded file payload size is enforced by checking the uploaded `model_file` stream against `MAX_MODEL_FILE_SIZE`.
- Multiple Gunicorn workers would each have independent model metadata and cache state.
