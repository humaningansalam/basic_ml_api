import logging
from lzma import LZMAError
import os
import re
import shutil
import tempfile
import threading
import time
import weakref
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional
from zipfile import BadZipFile, ZipFile, ZipInfo
from zlib import error as ZlibError

import numpy as np
import tensorflow as tf

from src.common import utils
from src.common.errors import ApplicationError, ErrorCode
from src.common.metrics import get_metrics
from src.core.model_types import CleanupResult, ModelInfo, ModelMetadata, PredictionResult, UploadResult


class ModelManager:
    _MODEL_HASH_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
    _BACKUP_PREFIX = '@backup-'
    _STAGING_PREFIX = '@upload-'
    _MAX_ZIP_ENTRIES = 5000

    def __init__(self, store_path: str, max_cache_size: int = 10, cleanup_interval_hours: Optional[int] = None, max_model_file_size: int = 100 * 1024 * 1024):
        self.store_path = store_path
        self.max_cache_size = max_cache_size
        self.max_model_file_size = max_model_file_size
        self.metadata_store: Dict[str, ModelMetadata] = {}
        self.model_cache = OrderedDict()
        self.metrics = get_metrics()
        self.logger = logging.getLogger(__name__)
        self.cleanup_interval_hours = cleanup_interval_hours if cleanup_interval_hours is not None else 5
        self._state_lock = threading.RLock()
        self._cleanup_locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
        self._cleanup_thread_started = False
        self._cleanup_thread_lock = threading.Lock()

        self._load_metadata_store()
        self.start_cleanup_scheduler()

    def _load_metadata_store(self) -> None:
        if not os.path.exists(self.store_path):
            os.makedirs(self.store_path)
            return

        self._recover_interrupted_uploads()

        for model_hash in os.listdir(self.store_path):
            model_folder_path = os.path.join(self.store_path, model_hash)
            if not self._MODEL_HASH_PATTERN.fullmatch(model_hash):
                continue
            if not os.path.isdir(model_folder_path):
                continue
            if not self._contains_keras_file(model_folder_path):
                self.logger.warning('Ignoring model directory without a .keras file: %s', model_folder_path)
                continue

            self.metadata_store[model_hash] = ModelMetadata(
                file_path=model_folder_path,
                used=utils.get_kr_time(),
            )

    def _contains_keras_file(self, model_folder_path: str) -> bool:
        for root, _, files in os.walk(model_folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith('.keras') and os.path.isfile(file_path):
                    return True
        return False

    def _backup_path(self, model_hash: str) -> str:
        return os.path.join(self.store_path, f'{self._BACKUP_PREFIX}{model_hash}')

    def _update_existing_metadata_path(self, model_hash: str, file_path: str) -> None:
        with self._state_lock:
            metadata = self.metadata_store.get(model_hash)
            if metadata is not None:
                metadata.file_path = file_path

    def _remove_path(self, path: str) -> None:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.exists(path) or os.path.islink(path):
            os.remove(path)

    def _recover_interrupted_uploads(self) -> None:
        for entry in os.listdir(self.store_path):
            if entry.startswith(self._STAGING_PREFIX):
                staging_path = os.path.join(self.store_path, entry)
                try:
                    self._remove_path(staging_path)
                except OSError as cleanup_error:
                    self.logger.warning(
                        'Could not remove interrupted upload workspace %s: %s',
                        staging_path,
                        cleanup_error,
                    )
                else:
                    self.logger.warning(
                        'Removed interrupted upload workspace: %s',
                        staging_path,
                    )
                continue

            if not entry.startswith(self._BACKUP_PREFIX):
                continue

            model_hash = entry[len(self._BACKUP_PREFIX):]
            if not self._MODEL_HASH_PATTERN.fullmatch(model_hash):
                continue

            backup_path = os.path.join(self.store_path, entry)
            model_folder_path = os.path.join(self.store_path, model_hash)
            if os.path.exists(model_folder_path):
                try:
                    self._remove_path(backup_path)
                except OSError as cleanup_error:
                    self.logger.warning('Could not remove leftover model backup %s: %s', backup_path, cleanup_error)
                else:
                    self.logger.warning('Removed leftover model backup after completed upload: %s', backup_path)
            else:
                os.replace(backup_path, model_folder_path)
                self.logger.warning('Restored model backup after interrupted upload: %s', model_hash)

    def _replace_model_directory(self, staging_dir: str, model_folder_path: str, model_hash: str) -> bool:
        backup_path = self._backup_path(model_hash)
        had_existing_model = os.path.exists(model_folder_path)

        if os.path.exists(backup_path) or os.path.islink(backup_path):
            if had_existing_model:
                self._remove_path(backup_path)
            else:
                os.replace(backup_path, model_folder_path)
                had_existing_model = True
                self._update_existing_metadata_path(model_hash, model_folder_path)

        if had_existing_model:
            os.replace(model_folder_path, backup_path)

        try:
            os.replace(staging_dir, model_folder_path)
        except OSError:
            if had_existing_model:
                try:
                    os.replace(backup_path, model_folder_path)
                except OSError as rollback_error:
                    for recovery_path in (backup_path, model_folder_path):
                        if self._contains_keras_file(recovery_path):
                            self._update_existing_metadata_path(model_hash, recovery_path)
                            break
                    raise OSError(
                        f'Failed to install model {model_hash} and restore its previous version'
                    ) from rollback_error
                self._update_existing_metadata_path(model_hash, model_folder_path)
            raise

        if had_existing_model:
            try:
                self._remove_path(backup_path)
            except OSError as cleanup_error:
                self.logger.warning('Could not remove model backup %s: %s', backup_path, cleanup_error)

        return had_existing_model

    def _validate_archive_members(self, members: list[ZipInfo], staging_dir: str) -> None:
        if len(members) > self._MAX_ZIP_ENTRIES:
            raise ApplicationError(
                ErrorCode.ZIP_ENTRY_LIMIT_EXCEEDED,
                {
                    'entry_count': len(members),
                    'max_entries': self._MAX_ZIP_ENTRIES,
                },
            )

        base_path = Path(staging_dir).resolve()
        archive_paths: Dict[PurePosixPath, bool] = {}
        has_keras = False
        total_uncompressed_size = 0

        for member_info in members:
            member = member_info.filename
            member_path = Path(member)
            archive_path = PurePosixPath(member)
            is_directory = member_info.is_dir()
            is_windows_drive_path = len(member) >= 2 and member[1] == ':'
            is_absolute_path = member_path.is_absolute() or member.startswith('\\') or is_windows_drive_path

            if is_absolute_path or any(part == '..' for part in member_path.parts):
                raise ApplicationError(ErrorCode.UNSAFE_ZIP_ENTRY, {'entry': member})

            target_path = (Path(staging_dir) / member_path).resolve()
            if not target_path.is_relative_to(base_path):
                raise ApplicationError(ErrorCode.UNSAFE_ZIP_ENTRY, {'entry': member})

            if archive_path == PurePosixPath('.') and not is_directory:
                raise ApplicationError(ErrorCode.INVALID_ZIP)

            previous_kind = archive_paths.get(archive_path)
            if previous_kind is not None and previous_kind != is_directory:
                raise ApplicationError(ErrorCode.INVALID_ZIP)
            archive_paths[archive_path] = is_directory

            if not is_directory and member.endswith('.keras'):
                has_keras = True

            total_uncompressed_size += member_info.file_size

        for archive_path in archive_paths:
            for parent in archive_path.parents:
                if parent == PurePosixPath('.'):
                    break
                if archive_paths.get(parent) is False:
                    raise ApplicationError(ErrorCode.INVALID_ZIP)

        if total_uncompressed_size > self.max_model_file_size:
            raise ApplicationError(
                ErrorCode.UNCOMPRESSED_SIZE_EXCEEDED,
                {
                    'expanded_bytes': total_uncompressed_size,
                    'max_bytes': self.max_model_file_size,
                },
            )

        if not has_keras:
            raise ApplicationError(ErrorCode.MODEL_ARTIFACT_REQUIRED)

    def _validate_archive_contents(self, zip_ref: ZipFile) -> None:
        # Decompress before writing so corrupt input stays distinct from storage failures.
        try:
            corrupt_member = zip_ref.testzip()
        except (BadZipFile, EOFError, LZMAError, OSError, RuntimeError, ZlibError) as error:
            raise ApplicationError(ErrorCode.INVALID_ZIP) from error

        if corrupt_member is not None:
            raise ApplicationError(ErrorCode.INVALID_ZIP)

    def start_cleanup_scheduler(self) -> None:
        with self._cleanup_thread_lock:
            if self._cleanup_thread_started:
                return
            if self.cleanup_interval_hours <= 0:
                return
            self._cleanup_thread_started = True

        def scheduled_cleanup():
            while True:
                try:
                    self.clean_old_models()
                except ApplicationError as error:
                    self.logger.error(
                        'Scheduled cleanup failed code=%s details=%s',
                        error.code.value,
                        dict(error.details),
                        exc_info=(type(error), error, error.__traceback__),
                    )
                time.sleep(self.cleanup_interval_hours * 3600)

        thread = threading.Thread(target=scheduled_cleanup, daemon=True)
        thread.start()

    def _get_model_dir_lock(self, model_hash: str) -> threading.RLock:
        with self._state_lock:
            lock = self._cleanup_locks.get(model_hash)
            if lock is None:
                lock = threading.RLock()
                self._cleanup_locks[model_hash] = lock
            return lock

    def clean_old_models(self) -> CleanupResult:
        stale_models = []
        removed_model_hashes = []
        with self._state_lock:
            cutoff = utils.one_week_ago()
            for model_hash, metadata in self.metadata_store.items():
                if metadata.used < cutoff:
                    stale_models.append((model_hash, metadata.file_path))

        for model_hash, model_path in stale_models:
            with self._get_model_dir_lock(model_hash):
                with self._state_lock:
                    current_metadata = self.metadata_store.get(model_hash)
                    if current_metadata is None:
                        continue
                    if current_metadata.used >= utils.one_week_ago():
                        continue
                    if current_metadata.file_path != model_path:
                        continue

                try:
                    backup_path = self._backup_path(model_hash)
                    if os.path.exists(backup_path) or os.path.islink(backup_path):
                        self._remove_path(backup_path)
                    if os.path.exists(model_path) or os.path.islink(model_path):
                        self._remove_path(model_path)
                except OSError as error:
                    raise ApplicationError(
                        ErrorCode.MODEL_STORAGE_FAILED,
                        {'model_hash': model_hash, 'operation': 'cleanup'},
                    ) from error

                with self._state_lock:
                    current_metadata = self.metadata_store.get(model_hash)
                    if current_metadata is None:
                        continue
                    if current_metadata.used >= utils.one_week_ago():
                        continue
                    if current_metadata.file_path != model_path:
                        continue

                    self.metadata_store.pop(model_hash, None)
                    if model_hash in self.model_cache:
                        del self.model_cache[model_hash]
                        self.metrics.set_model_cache_usage(len(self.model_cache))

                removed_model_hashes.append(model_hash)
                self.logger.info('Removed old model: %s', model_hash)

        return CleanupResult(tuple(removed_model_hashes))

    def load_model_to_cache(self, model_hash: str) -> Any:
        with self._get_model_dir_lock(model_hash):
            with self._state_lock:
                if model_hash in self.model_cache:
                    self.model_cache.move_to_end(model_hash)
                    self.metrics.increment_cache_hit()
                    return self.model_cache[model_hash]

                if model_hash not in self.metadata_store:
                    raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {'model_hash': model_hash})

                model_folder_path = self.metadata_store[model_hash].file_path

            keras_file_path = None
            for root, _, files in os.walk(model_folder_path):
                for file in files:
                    if file.endswith('.keras'):
                        keras_file_path = os.path.join(root, file)
                        break
                if keras_file_path:
                    break

            if not keras_file_path:
                raise ApplicationError(
                    ErrorCode.MODEL_ARTIFACT_UNAVAILABLE,
                    {'model_hash': model_hash},
                )

            model = tf.keras.models.load_model(keras_file_path)

            with self._state_lock:
                cached_model = self.model_cache.get(model_hash)
                if cached_model is not None:
                    self.model_cache.move_to_end(model_hash)
                    self.metrics.increment_cache_hit()
                    return cached_model

                current_metadata = self.metadata_store.get(model_hash)
                if current_metadata is None:
                    raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {'model_hash': model_hash})
                if current_metadata.file_path != model_folder_path:
                    raise ApplicationError(
                        ErrorCode.MODEL_STATE_CONFLICT,
                        {'model_hash': model_hash},
                    )

                if len(self.model_cache) >= self.max_cache_size:
                    self.model_cache.popitem(last=False)
                self.model_cache[model_hash] = model
                self.metrics.increment_cache_miss()
                self.metrics.set_model_cache_usage(len(self.model_cache))
                return model

    def predict(self, model_hash: str, data: np.ndarray) -> PredictionResult:
        try:
            with self._get_model_dir_lock(model_hash):
                model = self.load_model_to_cache(model_hash)
                with self._state_lock:
                    metadata = self.metadata_store.get(model_hash)
                    if metadata is None:
                        raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {'model_hash': model_hash})
                    metadata.used = utils.get_kr_time()
            prediction = model.predict(data)
            return PredictionResult(model_hash=model_hash, values=prediction)
        except ApplicationError:
            raise
        except Exception as error:
            raise ApplicationError(
                ErrorCode.PREDICTION_FAILED,
                {'model_hash': model_hash},
            ) from error

    def upload_model(self, model_file, model_hash: str) -> UploadResult:
        if not self._MODEL_HASH_PATTERN.fullmatch(model_hash or ""):
            raise ApplicationError(ErrorCode.INVALID_MODEL_HASH, {'model_hash': model_hash})

        model_folder_path = os.path.join(self.store_path, model_hash)
        model_dir_lock = self._get_model_dir_lock(model_hash)

        try:
            with model_dir_lock:
                os.makedirs(self.store_path, exist_ok=True)
                with tempfile.TemporaryDirectory(
                    prefix=f'{self._STAGING_PREFIX}{model_hash}-',
                    dir=self.store_path,
                ) as upload_workspace:
                    staging_dir = os.path.join(upload_workspace, 'model')
                    os.makedirs(staging_dir)
                    temp_zip_path = os.path.join(upload_workspace, 'archive.zip')
                    model_file.save(temp_zip_path)

                    try:
                        try:
                            with ZipFile(temp_zip_path, 'r') as zip_ref:
                                members = zip_ref.infolist()
                                self._validate_archive_members(members, staging_dir)
                                self._validate_archive_contents(zip_ref)
                                zip_ref.extractall(staging_dir)
                        except (BadZipFile, EOFError, LZMAError, RuntimeError, UnicodeError, ZlibError) as error:
                            raise ApplicationError(ErrorCode.INVALID_ZIP) from error
                    finally:
                        if os.path.exists(temp_zip_path):
                            os.remove(temp_zip_path)

                    used_at = utils.get_kr_time()
                    replaced = self._replace_model_directory(
                        staging_dir,
                        model_folder_path,
                        model_hash,
                    )

                    with self._state_lock:
                        if model_hash in self.model_cache:
                            del self.model_cache[model_hash]
                            self.metrics.set_model_cache_usage(len(self.model_cache))

                        self.metadata_store[model_hash] = ModelMetadata(
                            file_path=model_folder_path,
                            used=used_at,
                        )
        except ApplicationError:
            raise
        except OSError as error:
            raise ApplicationError(
                ErrorCode.MODEL_STORAGE_FAILED,
                {'model_hash': model_hash},
            ) from error

        return UploadResult(model_hash=model_hash, replaced=replaced)

    def get_model_info(self, model_hash: str) -> ModelInfo:
        with self._state_lock:
            if model_hash not in self.metadata_store:
                raise ApplicationError(ErrorCode.MODEL_NOT_FOUND, {'model_hash': model_hash})
            metadata = self.metadata_store[model_hash]
            return ModelInfo(
                model_hash=model_hash,
                file_path=metadata.file_path,
                used=metadata.used,
            )
