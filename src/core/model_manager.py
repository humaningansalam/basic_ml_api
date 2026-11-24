import logging
import os
import re
import shutil
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol
from zipfile import BadZipFile

import numpy as np
import tensorflow as tf

from src.common import clock
from src.common.errors import ApplicationError, ErrorCode
from src.common.metrics import get_metrics
from src.core.model_archive import extract_model_archive
from src.core.model_types import (
    CleanupResult,
    ModelInfo,
    ModelMetadata,
    PredictionResult,
    UploadResult,
)


class ModelArchiveUpload(Protocol):
    def save(self, destination: str) -> None: ...


class PredictiveModel(Protocol):
    def predict(self, prediction_input: np.ndarray) -> Any: ...


def _load_keras_model(artifact_path: str) -> PredictiveModel:
    return tf.keras.models.load_model(artifact_path, compile=False, safe_mode=True)


class ModelManager:
    """Own model storage, cache, and lifecycle transitions."""

    _MODEL_HASH_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
    _BACKUP_PREFIX = "@backup-"
    _STAGING_PREFIX = "@upload-"

    def __init__(
        self,
        model_store_path: str,
        max_cached_models: int = 10,
        cleanup_interval_hours: int | None = None,
        max_model_file_size: int = 100 * 1024 * 1024,
    ):
        if max_cached_models < 1:
            raise ValueError("max_cached_models must be at least 1")
        if max_model_file_size < 1:
            raise ValueError("max_model_file_size must be at least 1")
        if cleanup_interval_hours is not None and cleanup_interval_hours < 0:
            raise ValueError("cleanup_interval_hours must not be negative")

        self.model_store_path = model_store_path
        self.max_cached_models = max_cached_models
        self.max_model_file_size = max_model_file_size
        self._metadata_by_hash: dict[str, ModelMetadata] = {}
        self._loaded_models_by_hash: OrderedDict[str, PredictiveModel] = OrderedDict()
        self._metrics = get_metrics()
        self._logger = logging.getLogger(__name__)
        self.cleanup_interval_hours = (
            cleanup_interval_hours if cleanup_interval_hours is not None else 5
        )
        self._state_lock = threading.RLock()
        self._model_locks: weakref.WeakValueDictionary[str, threading.RLock] = (
            weakref.WeakValueDictionary()
        )
        self._cleanup_scheduler_started = False
        self._cleanup_scheduler_lock = threading.Lock()

        self._load_existing_models()
        self.start_cleanup_scheduler()

    def _load_existing_models(self) -> None:
        if not os.path.exists(self.model_store_path):
            os.makedirs(self.model_store_path, exist_ok=True)
            return

        self._recover_interrupted_uploads()

        for model_hash in sorted(os.listdir(self.model_store_path)):
            model_directory = os.path.join(self.model_store_path, model_hash)
            if not self._MODEL_HASH_PATTERN.fullmatch(model_hash):
                continue
            if not os.path.isdir(model_directory) or os.path.islink(model_directory):
                continue
            if self._find_model_artifact(model_directory) is None:
                self._logger.warning(
                    "Ignoring model directory without a .keras file: %s",
                    model_directory,
                )
                continue

            self._metadata_by_hash[model_hash] = ModelMetadata(
                file_path=model_directory,
                last_used_at=clock.utc_now(),
            )

    def _find_model_artifact(self, model_directory: str) -> str | None:
        if os.path.islink(model_directory):
            return None

        for current_directory, directories, files in os.walk(model_directory):
            directories[:] = sorted(
                directory_name
                for directory_name in directories
                if not os.path.islink(os.path.join(current_directory, directory_name))
            )
            for file_name in sorted(files):
                file_path = os.path.join(current_directory, file_name)
                if (
                    file_name.endswith(".keras")
                    and os.path.isfile(file_path)
                    and not os.path.islink(file_path)
                ):
                    return file_path
        return None

    def _replacement_backup_path(self, model_hash: str) -> str:
        return os.path.join(self.model_store_path, f"{self._BACKUP_PREFIX}{model_hash}")

    def _update_registered_model_path(self, model_hash: str, file_path: str) -> None:
        with self._state_lock:
            metadata = self._metadata_by_hash.get(model_hash)
            if metadata is not None:
                metadata.file_path = file_path

    def _remove_path(self, path: str) -> None:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.exists(path) or os.path.islink(path):
            os.remove(path)

    def _recover_interrupted_uploads(self) -> None:
        for entry in sorted(os.listdir(self.model_store_path)):
            if entry.startswith(self._STAGING_PREFIX):
                staging_path = os.path.join(self.model_store_path, entry)
                try:
                    self._remove_path(staging_path)
                except OSError as cleanup_error:
                    self._logger.warning(
                        "Could not remove interrupted upload workspace %s: %s",
                        staging_path,
                        cleanup_error,
                    )
                else:
                    self._logger.warning(
                        "Removed interrupted upload workspace: %s",
                        staging_path,
                    )
                continue

            if not entry.startswith(self._BACKUP_PREFIX):
                continue

            model_hash = entry[len(self._BACKUP_PREFIX) :]
            if not self._MODEL_HASH_PATTERN.fullmatch(model_hash):
                continue

            backup_path = os.path.join(self.model_store_path, entry)
            model_directory = os.path.join(self.model_store_path, model_hash)
            if os.path.exists(model_directory):
                try:
                    self._remove_path(backup_path)
                except OSError as cleanup_error:
                    self._logger.warning(
                        "Could not remove leftover model backup %s: %s",
                        backup_path,
                        cleanup_error,
                    )
                else:
                    self._logger.warning(
                        "Removed leftover model backup after completed upload: %s",
                        backup_path,
                    )
            else:
                os.replace(backup_path, model_directory)
                self._logger.warning(
                    "Restored model backup after interrupted upload: %s", model_hash
                )

    def _install_staged_model(
        self, staged_model_directory: str, model_directory: str, model_hash: str
    ) -> bool:
        backup_path = self._replacement_backup_path(model_hash)
        had_existing_model = os.path.exists(model_directory)

        if os.path.exists(backup_path) or os.path.islink(backup_path):
            if had_existing_model:
                self._remove_path(backup_path)
            else:
                os.replace(backup_path, model_directory)
                had_existing_model = True
                self._update_registered_model_path(model_hash, model_directory)

        if had_existing_model:
            os.replace(model_directory, backup_path)

        try:
            os.replace(staged_model_directory, model_directory)
        except OSError:
            if had_existing_model:
                try:
                    os.replace(backup_path, model_directory)
                except OSError as rollback_error:
                    for recovery_path in (backup_path, model_directory):
                        if self._find_model_artifact(recovery_path) is not None:
                            self._update_registered_model_path(model_hash, recovery_path)
                            break
                    raise OSError(
                        f"Failed to install model {model_hash} and restore its previous version"
                    ) from rollback_error
                self._update_registered_model_path(model_hash, model_directory)
            raise

        if had_existing_model:
            try:
                self._remove_path(backup_path)
            except OSError as cleanup_error:
                self._logger.warning(
                    "Could not remove model backup %s: %s", backup_path, cleanup_error
                )

        return had_existing_model

    def start_cleanup_scheduler(self) -> None:
        with self._cleanup_scheduler_lock:
            if self._cleanup_scheduler_started:
                return
            if self.cleanup_interval_hours <= 0:
                return
            self._cleanup_scheduler_started = True

        def scheduled_cleanup() -> None:
            while True:
                try:
                    self.remove_stale_models()
                except ApplicationError as error:
                    self._logger.error(
                        "Scheduled cleanup failed code=%s details=%s",
                        error.code.value,
                        dict(error.details),
                        exc_info=(type(error), error, error.__traceback__),
                    )
                except Exception:
                    self._logger.exception("Scheduled cleanup stopped by an unexpected error")
                time.sleep(self.cleanup_interval_hours * 3600)

        cleanup_thread = threading.Thread(
            target=scheduled_cleanup,
            name="model-cleanup",
            daemon=True,
        )
        cleanup_thread.start()

    def _get_model_lock(self, model_hash: str) -> threading.RLock:
        with self._state_lock:
            model_lock = self._model_locks.get(model_hash)
            if model_lock is None:
                model_lock = threading.RLock()
                self._model_locks[model_hash] = model_lock
            return model_lock

    def remove_stale_models(self) -> CleanupResult:
        stale_model_directories: list[tuple[str, str]] = []
        removed_model_hashes: list[str] = []
        with self._state_lock:
            stale_cutoff = clock.stale_model_cutoff()
            for model_hash, metadata in self._metadata_by_hash.items():
                if metadata.last_used_at < stale_cutoff:
                    stale_model_directories.append((model_hash, metadata.file_path))

        for model_hash, stale_model_directory in stale_model_directories:
            with self._get_model_lock(model_hash):
                with self._state_lock:
                    current_metadata = self._metadata_by_hash.get(model_hash)
                    if current_metadata is None:
                        continue
                    if current_metadata.last_used_at >= stale_cutoff:
                        continue
                    if current_metadata.file_path != stale_model_directory:
                        continue

                try:
                    backup_path = self._replacement_backup_path(model_hash)
                    if os.path.exists(backup_path) or os.path.islink(backup_path):
                        self._remove_path(backup_path)
                    if os.path.exists(stale_model_directory) or os.path.islink(
                        stale_model_directory
                    ):
                        self._remove_path(stale_model_directory)
                except OSError as error:
                    raise ApplicationError(
                        ErrorCode.MODEL_STORAGE_FAILED,
                        {"model_hash": model_hash, "operation": "cleanup"},
                    ) from error

                with self._state_lock:
                    self._metadata_by_hash.pop(model_hash, None)
                    if model_hash in self._loaded_models_by_hash:
                        del self._loaded_models_by_hash[model_hash]
                        self._metrics.set_loaded_model_count(len(self._loaded_models_by_hash))

                removed_model_hashes.append(model_hash)
                self._logger.info("Removed old model: %s", model_hash)

        return CleanupResult(tuple(removed_model_hashes))

    def _validate_model_artifact(self, model_directory: str, model_hash: str) -> None:
        artifact_path = self._find_model_artifact(model_directory)
        if artifact_path is None:
            raise ApplicationError(ErrorCode.MODEL_ARTIFACT_REQUIRED)
        try:
            _load_keras_model(artifact_path)
        except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile) as error:
            raise ApplicationError(
                ErrorCode.INVALID_MODEL_ARTIFACT,
                {"model_hash": model_hash},
            ) from error

    def _get_or_load_model(self, model_hash: str) -> PredictiveModel:
        with self._get_model_lock(model_hash):
            with self._state_lock:
                cached_model = self._loaded_models_by_hash.get(model_hash)
                if cached_model is not None:
                    self._loaded_models_by_hash.move_to_end(model_hash)
                    self._metrics.increment_cache_hit()
                    return cached_model

                metadata = self._metadata_by_hash.get(model_hash)
                if metadata is None:
                    raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {"model_hash": model_hash})

                model_directory = metadata.file_path

            keras_file_path = self._find_model_artifact(model_directory)
            if keras_file_path is None:
                raise ApplicationError(
                    ErrorCode.MODEL_ARTIFACT_UNAVAILABLE,
                    {"model_hash": model_hash},
                )

            loaded_model = _load_keras_model(keras_file_path)

            with self._state_lock:
                cached_model = self._loaded_models_by_hash.get(model_hash)
                if cached_model is not None:
                    self._loaded_models_by_hash.move_to_end(model_hash)
                    self._metrics.increment_cache_hit()
                    return cached_model

                current_metadata = self._metadata_by_hash.get(model_hash)
                if current_metadata is None:
                    raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {"model_hash": model_hash})
                if current_metadata.file_path != model_directory:
                    raise ApplicationError(
                        ErrorCode.MODEL_STATE_CONFLICT,
                        {"model_hash": model_hash},
                    )

                if len(self._loaded_models_by_hash) >= self.max_cached_models:
                    self._loaded_models_by_hash.popitem(last=False)
                self._loaded_models_by_hash[model_hash] = loaded_model
                self._metrics.increment_cache_miss()
                self._metrics.set_loaded_model_count(len(self._loaded_models_by_hash))
                return loaded_model

    def predict(self, model_hash: str, prediction_input: np.ndarray) -> PredictionResult:
        try:
            with self._get_model_lock(model_hash):
                prediction_model = self._get_or_load_model(model_hash)
                with self._state_lock:
                    metadata = self._metadata_by_hash.get(model_hash)
                    if metadata is None:
                        raise ApplicationError(
                            ErrorCode.MODEL_NOT_FOUND, {"model_hash": model_hash}
                        )
                    metadata.last_used_at = clock.utc_now()
            prediction_output = prediction_model.predict(prediction_input)
            return PredictionResult(model_hash=model_hash, values=prediction_output)
        except ApplicationError:
            raise
        except Exception as error:
            raise ApplicationError(
                ErrorCode.PREDICTION_FAILED,
                {"model_hash": model_hash},
            ) from error

    def upload_model(self, model_archive: ModelArchiveUpload, model_hash: str) -> UploadResult:
        if not self._MODEL_HASH_PATTERN.fullmatch(model_hash or ""):
            raise ApplicationError(ErrorCode.INVALID_MODEL_HASH, {"model_hash": model_hash})

        model_directory = os.path.join(self.model_store_path, model_hash)
        model_lock = self._get_model_lock(model_hash)

        try:
            with model_lock:
                os.makedirs(self.model_store_path, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f"{self._STAGING_PREFIX}{model_hash}-",
                    dir=self.model_store_path,
                ) as upload_workspace:
                    staged_model_directory = os.path.join(upload_workspace, "model")
                    os.makedirs(staged_model_directory)
                    uploaded_archive_path = os.path.join(upload_workspace, "archive.zip")
                    model_archive.save(uploaded_archive_path)

                    try:
                        extract_model_archive(
                            uploaded_archive_path,
                            staged_model_directory,
                            max_uncompressed_bytes=self.max_model_file_size,
                        )
                    finally:
                        Path(uploaded_archive_path).unlink(missing_ok=True)

                    self._validate_model_artifact(staged_model_directory, model_hash)
                    last_used_at = clock.utc_now()
                    replaced = self._install_staged_model(
                        staged_model_directory,
                        model_directory,
                        model_hash,
                    )

                    with self._state_lock:
                        if self._loaded_models_by_hash.pop(model_hash, None) is not None:
                            self._metrics.set_loaded_model_count(len(self._loaded_models_by_hash))

                        self._metadata_by_hash[model_hash] = ModelMetadata(
                            file_path=model_directory,
                            last_used_at=last_used_at,
                        )
        except ApplicationError:
            raise
        except OSError as error:
            raise ApplicationError(
                ErrorCode.MODEL_STORAGE_FAILED,
                {"model_hash": model_hash},
            ) from error

        return UploadResult(model_hash=model_hash, replaced=replaced)

    def get_model_info(self, model_hash: str) -> ModelInfo:
        with self._state_lock:
            metadata = self._metadata_by_hash.get(model_hash)
            if metadata is None:
                raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {"model_hash": model_hash})
            return ModelInfo(
                model_hash=model_hash,
                file_path=metadata.file_path,
                last_used_at=metadata.last_used_at,
            )
