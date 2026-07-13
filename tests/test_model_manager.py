from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.model_manager import ModelManager


def _register_model(manager, store_path, model_hash, used):
    model_dir = Path(store_path) / model_hash
    model_dir.mkdir()
    (model_dir / 'model.keras').write_bytes(b'model')
    manager.metadata_store[model_hash] = {
        'file_path': str(model_dir),
        'used': used,
    }


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
        manager.clean_old_models()

    assert not (tmp_path / model_hash).exists()
    assert model_hash not in manager.metadata_store
    assert model_hash not in manager.model_cache
