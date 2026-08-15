# Changelog

All notable changes to Basic ML API are documented in this file.

## [0.3.1] - 2026-08-15

### Changed

- Simplified model loading, cache, cleanup, request parsing, and error-metric internals without changing the public API.
- Added a Docker healthcheck backed by the existing `/health` endpoint.

## [0.3.0] - 2026-07-19

### Added

- Added a runnable model-serving quickstart with a deterministic demo Keras archive and verified upload, lookup, prediction, health, and Prometheus flows.
- Added stable JSON success and error contracts with enum-backed error codes, structured model metadata, replacement status, and nested model output serialization.
- Added regression coverage for archive safety, request validation, cache and cleanup concurrency, replacement recovery, restart behavior, and framework-level errors.

### Changed

- Model uploads now use bounded request and archive limits, isolated staging workspaces, atomic backup-and-rename replacement, and startup recovery for interrupted operations.
- Model loading now coordinates per hash, maintains an LRU cache without retaining rejected hashes, and removes stale active and backup artifacts together.
- Production images exclude development dependencies, CI tests a development-enabled image, and Gunicorn disables its unused control socket to keep container restarts reliable.
- Documentation now describes the current API contracts, exact metric names, deployment behavior, and operator workflow.

### Fixed

- Preserved the previous usable model and correct `replaced` result across failed installs, failed rollbacks, retries, and restarts.
- Classified corrupt or malformed ZIP archives, unsafe paths, duplicate size accounting, oversized payloads, and missing model artifacts with stable client-facing errors.
- Rejected malformed, non-standard numeric, ragged, excessively nested, and non-array prediction input without leaking internal failures.
- Rejected non-finite or unsupported prediction output, restored transport-level request bounds, and cleaned abandoned upload workspaces after interrupted processes.
