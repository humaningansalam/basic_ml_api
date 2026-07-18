import threading
from datetime import datetime
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np

from src.core.model_manager import ModelManager
from src.core.model_types import ModelMetadata


def test_predict_releases_lock_while_model_load_is_blocked(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    manager.metadata_store['testhash123'] = ModelMetadata(
        file_path=str(tmp_path / 'testhash123'),
        used=datetime(2024, 4, 27, 12, 0),
    )
    manager.model_cache = OrderedDict()

    load_started = threading.Event()
    release_load = threading.Event()
    info_result = {}

    def fake_load_model(path):
        load_started.set()
        release_load.wait(timeout=5)
        model = MagicMock()
        model.predict.return_value = np.array([[0.9, 0.1]])
        return model

    with patch('src.core.model_manager.os.walk', return_value=[(str(tmp_path), [], ['model.keras'])]), \
         patch('src.core.model_manager.tf.keras.models.load_model', side_effect=fake_load_model):
        prediction = {}

        def run_predict():
            prediction['value'] = manager.predict('testhash123', np.array([[1.0, 2.0]]))

        thread = threading.Thread(target=run_predict)
        thread.start()

        assert load_started.wait(timeout=5)

        def read_state():
            info_result['value'] = manager.get_model_info('testhash123')

        reader = threading.Thread(target=read_state)
        reader.start()
        reader.join(timeout=5)

        assert 'value' in info_result
        assert info_result['value'].file_path == str(tmp_path / 'testhash123')

        release_load.set()
        thread.join(timeout=5)

    assert prediction['value'].values == [[0.9, 0.1]]


def test_predict_blocks_stale_cleanup_until_used_timestamp_refreshes(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir()
    (model_dir / 'model.keras').write_bytes(b'model')
    manager.metadata_store[model_hash] = ModelMetadata(
        file_path=str(model_dir),
        used=datetime(2024, 1, 1),
    )
    manager.model_cache = OrderedDict()

    stale_cutoff = datetime(2025, 1, 1)
    fresh_used = datetime(2025, 2, 1)
    load_started = threading.Event()
    release_load = threading.Event()
    cleanup_lock_requested = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_result = {}
    prediction = {}

    def fake_load_model(path):
        load_started.set()
        release_load.wait(timeout=5)
        model = MagicMock()
        model.predict.return_value = np.array([[0.9, 0.1]])
        return model

    original_get_model_dir_lock = manager._get_model_dir_lock

    def tracked_get_model_dir_lock(current_hash):
        if threading.current_thread().name == 'cleanup-thread':
            cleanup_lock_requested.set()
        return original_get_model_dir_lock(current_hash)

    def run_predict():
        prediction['value'] = manager.predict(model_hash, np.array([[1.0, 2.0]]))

    def run_cleanup():
        cleanup_result['value'] = manager.clean_old_models()
        cleanup_finished.set()

    with patch('src.core.model_manager.os.walk', return_value=[(str(model_dir), [], ['model.keras'])]), \
         patch('src.core.model_manager.tf.keras.models.load_model', side_effect=fake_load_model), \
         patch('src.core.model_manager.utils.one_week_ago', return_value=stale_cutoff), \
         patch('src.core.model_manager.utils.get_kr_time', return_value=fresh_used), \
         patch.object(manager, '_get_model_dir_lock', side_effect=tracked_get_model_dir_lock):
        predict_thread = threading.Thread(target=run_predict)
        predict_thread.start()

        assert load_started.wait(timeout=5)

        cleanup_thread = threading.Thread(target=run_cleanup, name='cleanup-thread')
        cleanup_thread.start()

        assert cleanup_lock_requested.wait(timeout=5)
        assert not cleanup_finished.wait(timeout=0.1)

        release_load.set()
        predict_thread.join(timeout=5)
        cleanup_thread.join(timeout=5)

    assert cleanup_finished.is_set()
    assert cleanup_result['value'].removed_model_hashes == ()
    assert prediction['value'].values == [[0.9, 0.1]]
    assert (model_dir / 'model.keras').exists()
    assert manager.metadata_store[model_hash].used == fresh_used


def test_upload_and_cleanup_coordinate_on_same_model_directory(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir()
    (model_dir / 'stale.keras').write_bytes(b'stale-model')
    manager.metadata_store[model_hash] = ModelMetadata(
        file_path=str(model_dir),
        used=datetime(2024, 4, 27, 12, 0),
    )

    stale_cutoff = datetime(2025, 1, 1)
    fresh_used = datetime(2025, 2, 1)
    cleanup_paused = threading.Event()
    cleanup_may_continue = threading.Event()
    cleanup_finished = threading.Event()
    cleanup_result = {}
    metadata_snapshot = {}

    original_get_model_dir_lock = manager._get_model_dir_lock

    def revalidating_lock(current_hash):
        wrapped_lock = original_get_model_dir_lock(current_hash)

        class _WrappedLock:
            def __enter__(self):
                cleanup_paused.set()
                cleanup_may_continue.wait(timeout=5)
                return wrapped_lock.__enter__()

            def __exit__(self, exc_type, exc, tb):
                return wrapped_lock.__exit__(exc_type, exc, tb)

        return _WrappedLock()

    def fake_one_week_ago():
        return stale_cutoff

    def run_cleanup():
        cleanup_result['value'] = manager.clean_old_models()
        cleanup_finished.set()

    with patch('src.core.model_manager.utils.one_week_ago', side_effect=fake_one_week_ago), patch.object(manager, '_get_model_dir_lock', side_effect=revalidating_lock):
        cleanup_thread = threading.Thread(target=run_cleanup)
        cleanup_thread.start()

        assert cleanup_paused.wait(timeout=5)

        with manager._state_lock:
            manager.metadata_store[model_hash] = ModelMetadata(
                file_path=str(model_dir),
                used=fresh_used,
            )
            metadata_snapshot['value'] = manager.metadata_store[model_hash]
        (model_dir / 'model.keras').write_bytes(b'fresh-model')

        cleanup_may_continue.set()
        cleanup_thread.join(timeout=5)

    assert cleanup_finished.is_set()
    assert cleanup_result['value'].removed_model_hashes == ()
    assert (model_dir / 'model.keras').exists()
    assert manager.metadata_store[model_hash] == metadata_snapshot['value']


def test_cleanup_scheduler_start_is_idempotent_under_concurrency(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    manager._cleanup_thread_started = False
    manager.cleanup_interval_hours = 1
    real_thread = threading.Thread
    ready = threading.Barrier(5)
    thread_ctor_calls = []
    started_threads = []

    class _FakeSchedulerThread:
        def __init__(self, target=None, daemon=None):
            thread_ctor_calls.append((target, daemon))
            self._target = target
            self._daemon = daemon

        def start(self):
            started_threads.append((self._target, self._daemon))

    def fake_thread(target=None, daemon=None):
        return _FakeSchedulerThread(target=target, daemon=daemon)

    with patch('src.core.model_manager.threading.Thread', side_effect=fake_thread):
        def start_scheduler():
            ready.wait(timeout=5)
            manager.start_cleanup_scheduler()

        caller_threads = [real_thread(target=start_scheduler) for _ in range(5)]
        for caller in caller_threads:
            caller.start()
        for caller in caller_threads:
            caller.join(timeout=5)
            assert not caller.is_alive()

    assert len(thread_ctor_calls) == 1
    assert thread_ctor_calls[0][1] is True
    assert len(started_threads) == 1
    assert started_threads[0][1] is True


def test_cleanup_scheduler_can_be_started_explicitly(tmp_path):
    with patch('src.core.model_manager.threading.Thread') as mock_thread:
        manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
        assert mock_thread.call_count == 0

        manager.start_cleanup_scheduler()
        assert mock_thread.call_count == 0

        manager.cleanup_interval_hours = 1
        manager.start_cleanup_scheduler()
        assert mock_thread.call_count == 1

        manager.start_cleanup_scheduler()
        assert mock_thread.call_count == 1
