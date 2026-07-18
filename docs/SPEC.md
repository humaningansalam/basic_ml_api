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

## Response Contracts

Successful JSON endpoints return a `data` object. Errors use one schema and a stable enum-backed code:

```json
{
  "error": {
    "code": "model_not_found",
    "message": "The requested model was not found.",
    "details": {"model_hash": "demo-model-v1"}
  }
}
```

Clients must branch on `error.code`, not on the human-readable `message`. Error metrics use the same code as the `type` label.
Unknown routes return `404` with code `route_not_found`; unsupported methods return `405` with code `method_not_allowed`.
Other framework-level HTTP rejections preserve their HTTP status, use code `http_error`, and include the status in `error.details.status`.

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
- Query parameter: `hash` (`8` to `128` ASCII letters, digits, `.`, `_`, or `-`)
- Uploaded file format: ZIP archive containing at least one `.keras` file

### Success Response

- Status: `200`
- Body: `{"data": {"model_hash": "...", "replaced": false}}`

### Error Behavior

- Missing `model_file`: `400`, code `model_file_required`
- Missing `hash`: `400`, code `model_hash_required`
- Invalid hash: `400`, code `invalid_model_hash`
- Uploaded payload exceeds `MAX_MODEL_FILE_SIZE`: `413`, code `upload_too_large`
- Invalid ZIP: `400`, code `invalid_zip`
- Too many ZIP entries: `400`, code `zip_entry_limit_exceeded`
- Unsafe ZIP path: `400`, code `unsafe_zip_entry`
- Expanded archive exceeds `MAX_MODEL_FILE_SIZE`: `400`, code `uncompressed_size_exceeded`
- Missing `.keras`: `400`, code `model_artifact_required`
- Storage failure: `500`, code `model_storage_failed`; internal exception text is not exposed

### Storage Behavior

- The service creates a directory under `MODEL_STORE_PATH` named by the supplied hash.
- The uploaded ZIP is kept outside the extraction directory inside a temporary upload workspace, so archive member names cannot overwrite the service's own upload file.
- ZIP contents are first validated and extracted into a dedicated staging subdirectory.
- The target hash directory is replaced through a same-filesystem backup-and-rename transaction so stale files are removed without discarding the last working model first.
- If installing the staged directory fails, the previous directory is restored and its metadata and cache entry remain usable.
- If the process stops during replacement, startup restores an unfinished backup or removes a leftover backup after a completed install.
- The temporary uploaded ZIP is removed after the extraction attempt.
- Flask bounds the total request body to `MAX_MODEL_FILE_SIZE` plus 1 MiB of multipart overhead, while the upload endpoint separately enforces the exact `model_file` stream limit.
- If `hash` already exists, the old in-memory cached model is invalidated so the next `/predict` for that hash loads from disk again.
- If a hash is uploaded again, the existing directory contents are replaced and only the new upload is retained.
- Model metadata is stored in memory with `file_path` and `used` timestamp.
- Uploads larger than `MAX_MODEL_FILE_SIZE` are rejected with `413` and code `upload_too_large`.
- On startup, abandoned internal `@upload-...` workspaces are removed so interrupted uploads do not accumulate on the persistent volume; only valid hash directories containing a regular `.keras` file are registered, and other incomplete directories are ignored.

## Model Lookup

### Endpoint

`GET /get_model?hash={model_hash}`

### Success Response

- Status: `200`
- Body: `{"data": {"model_hash": "...", "file_path": "...", "used": "ISO-8601 timestamp"}}`

### Error Behavior

- Missing `hash`: `400`, code `model_hash_required`
- Unknown hash: `404`, code `model_not_found`
- Unexpected failure: `500`, code `internal_error`; internal exception text is not exposed

## Prediction

### Endpoint

`POST /predict?hash={model_hash}`

### Request

- Content type: `application/json`
- Query parameter: `hash`
- Body: JSON array-like prediction input data

### Success Response

- Status: `200`
- Body: `{"data": {"model_hash": "...", "prediction": [...]}}`
- Prediction output is converted from NumPy/TensorFlow output to JSON lists.
- Multi-output list or object structures preserve their nesting, with each NumPy/TensorFlow leaf converted to JSON-compatible values.

### Error Behavior

- Missing `hash`: `400`, code `model_hash_required`
- Missing JSON body: `400`, code `prediction_data_required`
- Malformed JSON: `400`, code `malformed_json`
- Non-array JSON: `400`, code `prediction_data_not_array`
- Empty array: `400`, code `prediction_data_empty`
- Ragged or otherwise invalid array shape: `400`, code `invalid_prediction_data`
- Unknown model hash: `404`, code `model_not_found`
- Missing stored artifact: `500`, code `model_artifact_unavailable`
- Unsupported or non-finite model output: `500`, code `prediction_failed`
- TensorFlow loading or prediction failure: `500`, code `prediction_failed`

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
- Per-hash coordination locks are retained only while operations use them, so rejected or deleted hashes do not accumulate process state.

## Stale Model Cleanup

- A background daemon thread runs cleanup on each `ModelManager` instance.
- Cleanup removes models whose `used` timestamp is older than one week.
- Cleanup removes the model directory and any deferred replacement backup before removing metadata and cached model instances, so a later restart cannot restore stale data.
- Cleanup interval is configurable through `MODEL_CLEANUP_INTERVAL`.

## Health Endpoint

### Endpoint

`GET /health`

### Success Response

- Status: `200`
- Body: `{"data": {"status": "healthy"}}`

## Metrics Endpoint

### Endpoint

`GET /metrics`

### Response

- Status: `200`
- Content type: Prometheus text format

### Expected Metrics

- `model_cache_usage`
- `predictions_completed_total`
- `cache_hits_total`
- `cache_misses_total`
- `ml_api_errors_total`
- `ml_api_cpu_usage_percent`
- `ml_api_ram_usage_mb`

## Deployment Notes

- The Docker image runs Gunicorn with one worker and disables the unused control socket: `gunicorn --no-control-socket -w 1 -b 0.0.0.0:5000 src.main:create_app()`.
- Docker Compose maps host port `12021` to container port `5000`.
- Docker Compose persists model data through `./data:/usr/src/app/data`.
- Docker Compose defaults image tags to `dev` when `VERSION` is unset and still accepts a local `.env` override.
- Logging and resource monitoring are initialized during app creation.

## Current Test Coverage

The test suite currently covers:

- Deterministic demo-model archive creation and real Keras loading.
- Health and metrics endpoint responses, including exact Prometheus metric names.
- Model upload success plus hash, payload-size, expanded-size, ZIP-entry, traversal, upload-workspace collision, and missing-model validation.
- Atomic replacement, cache invalidation, interrupted-upload recovery, and restart directory filtering.
- Prediction success plus missing, empty, malformed, non-array, and unknown-model requests.
- Model lookup success, missing hash, and unknown model behavior.
- LRU eviction, stale-model deletion, cleanup scheduling, and prediction/upload cleanup coordination.

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
