import logging
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zipfile import BadZipFile, ZipFile

import numpy as np
import tensorflow as tf

from src.common import utils
from src.common.metrics import get_metrics


class ModelManager:
    _MODEL_HASH_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,128}$")
    _STALE_MODEL_AGE_SECONDS = 7 * 24 * 60 * 60
    _BACKUP_PREFIX = '@backup-'
    _STAGING_PREFIX = '@upload-'

    def __init__(self, store_path: str, max_cache_size: int = 10, cleanup_interval_hours: Optional[int] = None, max_model_file_size: int = 100 * 1024 * 1024):
        self.store_path = store_path
        self.max_cache_size = max_cache_size
        self.max_model_file_size = max_model_file_size
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.model_cache = OrderedDict()
        self.metrics = get_metrics()
        self.logger = logging.getLogger(__name__)
        self.cleanup_interval_hours = cleanup_interval_hours if cleanup_interval_hours is not None else 5
        self._state_lock = threading.RLock()
        self._cleanup_locks: Dict[str, threading.RLock] = {}
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

            self.metadata_store[model_hash] = {
                'file_path': model_folder_path,
                'used': utils.get_kr_time()
            }

    def _contains_keras_file(self, model_folder_path: str) -> bool:
        for root, _, files in os.walk(model_folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith('.keras') and os.path.isfile(file_path):
                    return True
        return False

    def _backup_path(self, model_hash: str) -> str:
        return os.path.join(self.store_path, f'{self._BACKUP_PREFIX}{model_hash}')

    def _remove_path(self, path: str) -> None:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.exists(path) or os.path.islink(path):
            os.remove(path)

    def _recover_interrupted_uploads(self) -> None:
        for entry in os.listdir(self.store_path):
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

    def _replace_model_directory(self, staging_dir: str, model_folder_path: str, model_hash: str) -> None:
        backup_path = self._backup_path(model_hash)
        had_existing_model = os.path.exists(model_folder_path)

        if os.path.exists(backup_path) or os.path.islink(backup_path):
            if had_existing_model:
                self._remove_path(backup_path)
            else:
                os.replace(backup_path, model_folder_path)
                had_existing_model = True

        if had_existing_model:
            os.replace(model_folder_path, backup_path)

        try:
            os.replace(staging_dir, model_folder_path)
        except OSError:
            if had_existing_model:
                try:
                    os.replace(backup_path, model_folder_path)
                except OSError as rollback_error:
                    raise OSError(
                        f'Failed to install model {model_hash} and restore its previous version'
                    ) from rollback_error
            raise

        if had_existing_model:
            try:
                self._remove_path(backup_path)
            except OSError as cleanup_error:
                self.logger.warning('Could not remove model backup %s: %s', backup_path, cleanup_error)

    def start_cleanup_scheduler(self) -> None:
        with self._cleanup_thread_lock:
            if self._cleanup_thread_started:
                return
            if self.cleanup_interval_hours <= 0:
                return
            self._cleanup_thread_started = True

        def scheduled_cleanup():
            while True:
                self.clean_old_models()
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

    def clean_old_models(self) -> None:
        try:
            stale_models = []
            with self._state_lock:
                cutoff = utils.one_week_ago()
                for model_hash, metadata in self.metadata_store.items():
                    if metadata['used'] < cutoff:
                        stale_models.append((model_hash, metadata['file_path']))

            for model_hash, model_path in stale_models:
                with self._get_model_dir_lock(model_hash):
                    with self._state_lock:
                        current_metadata = self.metadata_store.get(model_hash)
                        if current_metadata is None:
                            continue
                        if current_metadata['used'] >= utils.one_week_ago():
                            continue
                        if current_metadata['file_path'] != model_path:
                            continue

                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)

                    with self._state_lock:
                        current_metadata = self.metadata_store.get(model_hash)
                        if current_metadata is None:
                            continue
                        if current_metadata['used'] >= utils.one_week_ago():
                            continue
                        if current_metadata['file_path'] != model_path:
                            continue

                        self.metadata_store.pop(model_hash, None)
                        if model_hash in self.model_cache:
                            del self.model_cache[model_hash]
                            self.metrics.set_model_cache_usage(len(self.model_cache))

                    self.logger.info(f"Removed old model: {model_hash}")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def load_model_to_cache(self, model_hash: str) -> Optional[Any]:
        with self._get_model_dir_lock(model_hash):
            with self._state_lock:
                if model_hash in self.model_cache:
                    self.model_cache.move_to_end(model_hash)
                    self.metrics.increment_cache_hit()
                    return self.model_cache[model_hash]

                if model_hash not in self.metadata_store:
                    raise KeyError(f"Model hash {model_hash} not found")

                model_folder_path = self.metadata_store[model_hash]['file_path']

            keras_file_path = None
            for root, _, files in os.walk(model_folder_path):
                for file in files:
                    if file.endswith('.keras'):
                        keras_file_path = os.path.join(root, file)
                        break
                if keras_file_path:
                    break

            if not keras_file_path:
                raise OSError('No .keras file found')

            model = tf.keras.models.load_model(keras_file_path)

            with self._state_lock:
                cached_model = self.model_cache.get(model_hash)
                if cached_model is not None:
                    self.model_cache.move_to_end(model_hash)
                    self.metrics.increment_cache_hit()
                    return cached_model

                current_metadata = self.metadata_store.get(model_hash)
                if current_metadata is None:
                    raise KeyError(f"Model hash {model_hash} not found")
                if current_metadata['file_path'] != model_folder_path:
                    raise KeyError(f"Model hash {model_hash} changed during load")

                if len(self.model_cache) >= self.max_cache_size:
                    self.model_cache.popitem(last=False)
                self.model_cache[model_hash] = model
                self.metrics.increment_cache_miss()
                self.metrics.set_model_cache_usage(len(self.model_cache))
                return model

    def predict(self, model_hash: str, data: np.ndarray) -> Tuple[np.ndarray, int]:
        try:
            with self._get_model_dir_lock(model_hash):
                model = self.load_model_to_cache(model_hash)
                with self._state_lock:
                    metadata = self.metadata_store.get(model_hash)
                    if metadata is None:
                        raise KeyError(f"Model hash {model_hash} not found")
                    metadata['used'] = utils.get_kr_time()
            prediction = model.predict(data)
            return prediction, 200
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def upload_model(self, model_file, model_hash: str) -> Tuple[str, int]:
        if not self._MODEL_HASH_PATTERN.fullmatch(model_hash or ""):
            raise ValueError('Invalid model hash')

        model_folder_path = os.path.join(self.store_path, model_hash)
        model_dir_lock = self._get_model_dir_lock(model_hash)

        with model_dir_lock:
            os.makedirs(self.store_path, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f'{self._STAGING_PREFIX}{model_hash}-', dir=self.store_path) as staging_dir:
                temp_zip_path = os.path.join(staging_dir, 'temp.zip')
                model_file.save(temp_zip_path)

                try:
                    try:
                        with ZipFile(temp_zip_path, 'r') as zip_ref:
                            members = zip_ref.infolist()
                            base_path = Path(staging_dir).resolve()
                            has_keras = False
                            total_uncompressed_size = 0
                            max_entries = 5000

                            if len(members) > max_entries:
                                raise ValueError(f'Zip contains too many entries: {len(members)} (max {max_entries})')

                            for member_info in members:
                                member = member_info.filename
                                member_path = Path(member)
                                member_text = member.replace('\\', '/')
                                is_windows_drive_path = len(member) >= 2 and member[1] == ':'
                                is_absolute_path = member_path.is_absolute() or member.startswith('\\') or is_windows_drive_path

                                if is_absolute_path or any(part == '..' for part in member_path.parts):
                                    raise ValueError('Unsafe zip entry')

                                target_path = (Path(staging_dir) / member_path).resolve()
                                if not target_path.is_relative_to(base_path):
                                    raise ValueError('Unsafe zip entry')

                                if member_text.endswith('.keras'):
                                    has_keras = True

                                total_uncompressed_size += member_info.file_size

                            if total_uncompressed_size > self.max_model_file_size:
                                raise ValueError(f'Uncompressed size too large: {total_uncompressed_size} bytes')

                            if not has_keras:
                                raise ValueError('No .keras file in zip')

                            zip_ref.extractall(staging_dir)
                    except BadZipFile:
                        raise ValueError('Invalid zip file')
                finally:
                    if os.path.exists(temp_zip_path):
                        os.remove(temp_zip_path)

                used_at = utils.get_kr_time()
                self._replace_model_directory(staging_dir, model_folder_path, model_hash)

                with self._state_lock:
                    if model_hash in self.model_cache:
                        del self.model_cache[model_hash]
                        self.metrics.set_model_cache_usage(len(self.model_cache))

                    self.metadata_store[model_hash] = {
                        'file_path': model_folder_path,
                        'used': used_at
                    }

        return 'Model uploaded successfully', 200

    def get_model_info(self, model_hash: str) -> Dict[str, str]:
        with self._state_lock:
            if model_hash not in self.metadata_store:
                raise KeyError(f"Model {model_hash} not found")
            return dict(self.metadata_store[model_hash])
