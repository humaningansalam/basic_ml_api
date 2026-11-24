# Changelog

All notable changes to Basic ML API are documented in this file.

## [0.3.2] - 2026-11-24

### Added

- Added a dedicated model-archive boundary for ZIP validation and extraction.
- Added repository architecture, contribution, security, agent, and pull-request guidance.
- Added regression coverage for configuration limits, safe Keras loading, symbolic-link entries, and non-portable archive paths.

### Changed

- Renamed model lifecycle fields and functions to describe cache, metadata, timestamps, locks, and replacement behavior directly.
- Standardized internal time handling on timezone-aware UTC.
- Made cache size configurable and validated all integer settings at startup.
- Loaded prediction-only Keras models with `compile=False` and `safe_mode=True`.
- Registered application services through Flask extensions instead of ad-hoc application attributes.
- Split documentation ownership between usage, external contracts, and internal architecture.
- Replaced broad generated ignore rules and duplicated automation with project-specific configuration.
- Pinned the Git-based monitoring dependency to a commit and removed unused direct dependencies.
- Uses `tensorflow-cpu` on Linux x86-64 and `tensorflow` on ARM64 to retain native CPU-only deployment compatibility.
- Kept the existing Compose service name and host model directory; the deployment prepares ownership for the non-root runtime without moving model data.

### Fixed

- Rejects unloadable Keras artifacts before replacing an existing model, preserving its files, metadata, and cache.
- Uses an existing setup-uv action tag so the release workflow can run.

### Security

- Rejects ZIP symbolic links and backslash-based archive paths in addition to traversal, absolute paths, conflicting entries, and size limits.
- Documents that authentication and model sandboxing are outside the service boundary.

### Removed

- Removed the `pytz`-based local-time helper, duplicated pytest configuration, and generic generated repository boilerplate.

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
