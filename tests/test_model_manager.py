from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.common.errors import ApplicationError, ErrorCode
from src.core.model_manager import ModelManager
from src.core.model_types import ModelMetadata


def _register_model(manager, store_path, model_hash, used):
    model_dir = Path(store_path) / model_hash
    model_dir.mkdir()
    (model_dir / 'model.keras').write_bytes(b'model')
    manager.metadata_store[model_hash] = ModelMetadata(
        file_path=str(model_dir),
        used=used,
    )


def test_model_cache_evicts_least_recently_used_model(tmp_path):
    manager = ModelManager(str(tmp_path), max_cache_size=2, cleanup_interval_hours=0)
    loaded_models = {}
    for model_hash in ('model-one', 'model-two', 'model-three'):
        _register_model(manager, tmp_path, model_hash, datetime.now(timezone.utc))
        loaded_models[model_hash] = MagicMock(name=model_hash)

    def load_model(model_path):
        return loaded_models[Path(model_path).parent.name]

    with patch('src.core.model_manager.tf.keras.models.load_model', side_effect=load_model) as mock_load:
        assert manager.load_model_to_cache('model-one') is loaded_models['model-one']
        assert manager.load_model_to_cache('model-two') is loaded_models['model-two']
        assert manager.load_model_to_cache('model-one') is loaded_models['model-one']
        assert manager.load_model_to_cache('model-three') is loaded_models['model-three']

    assert list(manager.model_cache) == ['model-one', 'model-three']
    assert mock_load.call_count == 3


def test_clean_old_models_removes_disk_metadata_and_cache(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'stale-model'
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=timezone.utc))
    manager.model_cache[model_hash] = MagicMock()

    with patch(
        'src.core.model_manager.utils.one_week_ago',
        return_value=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ):
        result = manager.clean_old_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
    assert model_hash not in manager.metadata_store
    assert model_hash not in manager.model_cache


def test_clean_old_models_reports_storage_failure(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'stale-model'
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=timezone.utc))

    with patch(
        'src.core.model_manager.utils.one_week_ago',
        return_value=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ), patch('src.core.model_manager.shutil.rmtree', side_effect=OSError('disk failure')):
        with pytest.raises(ApplicationError) as error_info:
            manager.clean_old_models()

    assert error_info.value.code is ErrorCode.MODEL_STORAGE_FAILED
    assert error_info.value.details == {'model_hash': model_hash, 'operation': 'cleanup'}
    assert model_hash in manager.metadata_store


def test_clean_old_models_removes_deferred_backup_before_restart(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'stale-model'
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=timezone.utc))
    backup_dir = tmp_path / f'@backup-{model_hash}'
    backup_dir.mkdir()
    (backup_dir / 'old.keras').write_bytes(b'old model')

    with patch(
        'src.core.model_manager.utils.one_week_ago',
        return_value=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ):
        result = manager.clean_old_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
    assert not backup_dir.exists()

    restarted_manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    assert model_hash not in restarted_manager.metadata_store


def test_clean_old_models_preserves_active_model_when_backup_cleanup_fails(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'stale-model'
    model_dir = tmp_path / model_hash
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=timezone.utc))
    backup_dir = tmp_path / f'@backup-{model_hash}'
    backup_dir.mkdir()
    (backup_dir / 'old.keras').write_bytes(b'old model')
    real_remove_path = manager._remove_path

    def fail_backup_cleanup(path):
        if Path(path) == backup_dir:
            raise OSError('disk failure')
        return real_remove_path(path)

    with patch(
        'src.core.model_manager.utils.one_week_ago',
        return_value=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ), patch.object(manager, '_remove_path', side_effect=fail_backup_cleanup):
        with pytest.raises(ApplicationError) as error_info:
            manager.clean_old_models()

    assert error_info.value.code is ErrorCode.MODEL_STORAGE_FAILED
    assert model_dir.exists()
    assert backup_dir.exists()
    assert model_hash in manager.metadata_store


def test_clean_old_models_uses_exact_one_week_cutoff(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'stale-model'
    now = datetime(2026, 7, 15, 14, 0, tzinfo=timezone.utc)
    _register_model(
        manager,
        tmp_path,
        model_hash,
        now - timedelta(weeks=1, seconds=1),
    )

    with patch('src.common.utils.get_kr_time', return_value=now):
        result = manager.clean_old_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
