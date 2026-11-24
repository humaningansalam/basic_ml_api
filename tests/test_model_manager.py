from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.common.errors import ApplicationError, ErrorCode
from src.core.model_manager import ModelManager
from src.core.model_types import ModelMetadata


def _register_model(manager, model_store_path, model_hash, last_used_at):
    model_dir = Path(model_store_path) / model_hash
    model_dir.mkdir()
    (model_dir / "model.keras").write_bytes(b"model")
    manager._metadata_by_hash[model_hash] = ModelMetadata(
        file_path=str(model_dir),
        last_used_at=last_used_at,
    )


def test_model_cache_evicts_least_recently_used_model(tmp_path):
    manager = ModelManager(str(tmp_path), max_cached_models=2, cleanup_interval_hours=0)
    loaded_models = {}
    for model_hash in ("model-one", "model-two", "model-three"):
        _register_model(manager, tmp_path, model_hash, datetime.now(UTC))
        loaded_models[model_hash] = MagicMock(name=model_hash)

    def load_model(model_path, **load_options):
        return loaded_models[Path(model_path).parent.name]

    with patch(
        "src.core.model_manager.tf.keras.models.load_model", side_effect=load_model
    ) as mock_load:
        assert manager._get_or_load_model("model-one") is loaded_models["model-one"]
        assert manager._get_or_load_model("model-two") is loaded_models["model-two"]
        assert manager._get_or_load_model("model-one") is loaded_models["model-one"]
        assert manager._get_or_load_model("model-three") is loaded_models["model-three"]

    assert list(manager._loaded_models_by_hash) == ["model-one", "model-three"]
    assert mock_load.call_count == 3
    assert all(
        call.kwargs == {"compile": False, "safe_mode": True} for call in mock_load.call_args_list
    )


@pytest.mark.parametrize(
    ("keyword_arguments", "message"),
    [
        ({"max_cached_models": 0}, "max_cached_models must be at least 1"),
        ({"max_model_file_size": 0}, "max_model_file_size must be at least 1"),
        ({"cleanup_interval_hours": -1}, "cleanup_interval_hours must not be negative"),
    ],
)
def test_model_manager_rejects_invalid_limits(tmp_path, keyword_arguments, message):
    with pytest.raises(ValueError, match=message):
        ModelManager(str(tmp_path), **keyword_arguments)


def test_remove_stale_models_removes_disk_metadata_and_cache(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = "stale-model"
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=UTC))
    manager._loaded_models_by_hash[model_hash] = MagicMock()

    with patch(
        "src.core.model_manager.clock.stale_model_cutoff",
        return_value=datetime(2025, 1, 1, tzinfo=UTC),
    ):
        result = manager.remove_stale_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
    assert model_hash not in manager._metadata_by_hash
    assert model_hash not in manager._loaded_models_by_hash


def test_remove_stale_models_reports_storage_failure(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = "stale-model"
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=UTC))

    with (
        patch(
            "src.core.model_manager.clock.stale_model_cutoff",
            return_value=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        patch("src.core.model_manager.shutil.rmtree", side_effect=OSError("disk failure")),
        pytest.raises(ApplicationError) as error_info,
    ):
        manager.remove_stale_models()

    assert error_info.value.code is ErrorCode.MODEL_STORAGE_FAILED
    assert error_info.value.details == {
        "model_hash": model_hash,
        "operation": "cleanup",
    }
    assert model_hash in manager._metadata_by_hash


def test_remove_stale_models_removes_deferred_backup_before_restart(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = "stale-model"
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=UTC))
    backup_dir = tmp_path / f"@backup-{model_hash}"
    backup_dir.mkdir()
    (backup_dir / "old.keras").write_bytes(b"old model")

    with patch(
        "src.core.model_manager.clock.stale_model_cutoff",
        return_value=datetime(2025, 1, 1, tzinfo=UTC),
    ):
        result = manager.remove_stale_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
    assert not backup_dir.exists()

    restarted_manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    assert model_hash not in restarted_manager._metadata_by_hash


def test_remove_stale_models_preserves_active_model_when_backup_cleanup_fails(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = "stale-model"
    model_dir = tmp_path / model_hash
    _register_model(manager, tmp_path, model_hash, datetime(2024, 1, 1, tzinfo=UTC))
    backup_dir = tmp_path / f"@backup-{model_hash}"
    backup_dir.mkdir()
    (backup_dir / "old.keras").write_bytes(b"old model")
    real_remove_path = manager._remove_path

    def fail_backup_cleanup(path):
        if Path(path) == backup_dir:
            raise OSError("disk failure")
        return real_remove_path(path)

    with (
        patch(
            "src.core.model_manager.clock.stale_model_cutoff",
            return_value=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        patch.object(manager, "_remove_path", side_effect=fail_backup_cleanup),
        pytest.raises(ApplicationError) as error_info,
    ):
        manager.remove_stale_models()

    assert error_info.value.code is ErrorCode.MODEL_STORAGE_FAILED
    assert model_dir.exists()
    assert backup_dir.exists()
    assert model_hash in manager._metadata_by_hash


def test_remove_stale_models_uses_exact_one_week_cutoff(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = "stale-model"
    now = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)
    _register_model(
        manager,
        tmp_path,
        model_hash,
        now - timedelta(weeks=1, seconds=1),
    )

    with patch("src.core.model_manager.clock.utc_now", return_value=now):
        result = manager.remove_stale_models()

    assert result.removed_model_hashes == (model_hash,)
    assert not (tmp_path / model_hash).exists()
