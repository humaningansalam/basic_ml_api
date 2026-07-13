import pytest
import io
import os
import zipfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config import Config
from src.core.model_manager import ModelManager
from src.main import create_app

def create_test_model_zip():
    """테스트용 모델 ZIP 파일 생성 (keras 파일 포함)"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('model.keras', b'dummy content')
    memory_file.seek(0)
    return memory_file


def create_test_model_zip_bytes():
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('replacement-model.keras', b'replacement content')
    memory_file.seek(0)
    return memory_file.getvalue()


def create_zip_info(filename, file_size=100):
    zip_info = zipfile.ZipInfo(filename)
    zip_info.file_size = file_size
    return zip_info


class _UploadFile:
    def __init__(self, content):
        self._content = content

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(self._content)

@patch('werkzeug.datastructures.FileStorage.save')
@patch('src.core.model_manager.ZipFile')
def test_upload_model_success(mock_zipfile, mock_save, client):
    """모델 업로드 성공 테스트"""
    # ZipFile 동작 모킹
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.infolist.return_value = [create_zip_info('model.keras')]

    test_zip = create_test_model_zip()

    # 테스트 실행
    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (test_zip, 'model.zip')},
                         content_type='multipart/form-data')

    mock_save.assert_called_once()

    assert response.status_code == 200
    assert response.json['message'] == 'Model uploaded successfully'

def test_upload_model_missing_data(client, get_metric_value):
    """필수 데이터 누락 테스트"""
    response = client.post('/upload_model')
    
    assert response.status_code == 400
    assert 'Missing data' in response.json['error']
    
    counter_value = get_metric_value('ml_api_errors', {'type': 'upload_model_missing_data'})
    assert counter_value == 1

@patch('werkzeug.datastructures.FileStorage.save')
@patch('src.core.model_manager.ZipFile')
def test_upload_model_bad_zip_is_client_error(mock_zipfile, mock_save, client):
    from zipfile import BadZipFile

    mock_zipfile.side_effect = BadZipFile('bad zip')
    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (io.BytesIO(b'not-a-zip'), 'model.zip')},
                         content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json['error'] == 'Invalid zip file'


@patch('werkzeug.datastructures.FileStorage.save')
@patch('src.core.model_manager.ZipFile')
def test_upload_model_missing_keras_is_client_error(mock_zipfile, mock_save, client):
    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.infolist.return_value = [create_zip_info('model.txt')]
    mock_zipfile.return_value = mock_zip

    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (io.BytesIO(b'zip'), 'model.zip')},
                         content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json['error'] == 'No .keras file in zip'


@patch('werkzeug.datastructures.FileStorage.save')
@patch('src.core.model_manager.ZipFile')
def test_upload_model_rejects_traversal_entry(mock_zipfile, mock_save, client):
    mock_zip = MagicMock()
    mock_zip.__enter__.return_value = mock_zip
    mock_zip.infolist.return_value = [
        create_zip_info('../escape.txt'),
        create_zip_info('safe.keras'),
    ]
    mock_zipfile.return_value = mock_zip

    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (io.BytesIO(b'zip'), 'model.zip')},
                         content_type='multipart/form-data')

    assert response.status_code == 400
    assert response.json['error'] == 'Unsafe zip entry'


def test_upload_model_small_file_reaches_zip_validation(client):
    client.application.config['MAX_MODEL_FILE_SIZE'] = 100
    response = client.post(
        '/upload_model?hash=testhash123',
        data={'model_file': (io.BytesIO(b'a'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Invalid zip file'


def test_upload_model_small_file_limit_uses_file_payload(tmp_path):
    class TinyPayloadLimitConfig(Config):
        TESTING = True
        START_BACKGROUND_MONITORING = False
        MODEL_STORE_PATH = str(tmp_path / 'models')
        MODEL_CLEANUP_INTERVAL = 0
        MAX_MODEL_FILE_SIZE = 100

    app = create_app(TinyPayloadLimitConfig)
    response = app.test_client().post(
        '/upload_model?hash=testhash123',
        data={'model_file': (io.BytesIO(b'a'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Invalid zip file'


def test_upload_model_oversized_file_payload_is_rejected(client):
    client.application.config['MAX_MODEL_FILE_SIZE'] = 1
    response = client.post(
        '/upload_model?hash=testhash123',
        data={'model_file': (io.BytesIO(b'abc'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 413
    assert response.json['error'] == 'Uploaded file too large'


def test_upload_model_counts_duplicate_entries_toward_uncompressed_limit(tmp_path):
    class SmallArchiveLimitConfig(Config):
        TESTING = True
        START_BACKGROUND_MONITORING = False
        MODEL_STORE_PATH = str(tmp_path / 'models')
        MODEL_CLEANUP_INTERVAL = 0
        MAX_MODEL_FILE_SIZE = 512

    archive = io.BytesIO()
    with pytest.warns(UserWarning, match='Duplicate name'):
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr('model.keras', b'A' * 10_000)
            zip_file.writestr('model.keras', b'B')
    archive.seek(0)

    assert len(archive.getvalue()) < SmallArchiveLimitConfig.MAX_MODEL_FILE_SIZE

    app = create_app(SmallArchiveLimitConfig)
    response = app.test_client().post(
        '/upload_model?hash=testhash123',
        data={'model_file': (archive, 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Uncompressed size too large: 10001 bytes'
    assert not (Path(SmallArchiveLimitConfig.MODEL_STORE_PATH) / 'testhash123').exists()


@pytest.mark.parametrize(
    'model_hash',
    [
        'short',
        '../escape',
        'nested/path',
        'nested\\path',
        '/absolute/path',
        'C:/windows/path',
        'C:\\windows\\path',
    ],
)
def test_upload_model_rejects_unsafe_hashes(model_hash, client):
    response = client.post(
        f'/upload_model?hash={model_hash}',
        data={'model_file': (io.BytesIO(b'zip'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Invalid model hash'


@patch('werkzeug.datastructures.FileStorage.save')
def test_upload_model_rejects_unsafe_hash_before_save(mock_save, client):
    response = client.post(
        '/upload_model?hash=../escape',
        data={'model_file': (io.BytesIO(b'zip'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Invalid model hash'
    mock_save.assert_not_called()


@patch('werkzeug.datastructures.FileStorage.save')
def test_upload_model_rejects_empty_hash_before_save(mock_save, client):
    response = client.post(
        '/upload_model',
        data={'model_file': (io.BytesIO(b'zip'), 'model.zip')},
        content_type='multipart/form-data',
    )

    assert response.status_code == 400
    assert response.json['error'] == 'Missing data (file or hash)'
    mock_save.assert_not_called()


def test_model_manager_accepts_valid_existing_hash(tmp_path):
    manager = ModelManager(str(tmp_path))

    assert manager._MODEL_HASH_PATTERN.fullmatch('testhash123')


def test_model_manager_ignores_incomplete_and_internal_directories_on_restart(tmp_path):
    (tmp_path / 'testhash123-upload-crash').mkdir()
    hidden_staging = tmp_path / '@upload-testhash123-crash'
    hidden_staging.mkdir()
    (hidden_staging / 'model.keras').write_bytes(b'incomplete')
    invalid_hash = tmp_path / 'short'
    invalid_hash.mkdir()
    (invalid_hash / 'model.keras').write_bytes(b'invalid hash')
    directory_only_model = tmp_path / 'validhash123'
    (directory_only_model / 'fake.keras').mkdir(parents=True)

    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    assert manager.metadata_store == {}


def test_model_manager_restores_backup_after_interrupted_replacement(tmp_path):
    backup_dir = tmp_path / '@backup-testhash123'
    backup_dir.mkdir()
    (backup_dir / 'old.keras').write_bytes(b'old model')

    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    model_dir = tmp_path / 'testhash123'
    assert (model_dir / 'old.keras').read_bytes() == b'old model'
    assert not backup_dir.exists()
    assert manager.metadata_store['testhash123']['file_path'] == str(model_dir)


def test_model_manager_removes_backup_after_completed_replacement(tmp_path):
    model_dir = tmp_path / 'testhash123'
    model_dir.mkdir()
    (model_dir / 'new.keras').write_bytes(b'new model')
    backup_dir = tmp_path / '@backup-testhash123'
    backup_dir.mkdir()
    (backup_dir / 'old.keras').write_bytes(b'old model')

    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    assert (model_dir / 'new.keras').read_bytes() == b'new model'
    assert not backup_dir.exists()
    assert manager.metadata_store['testhash123']['file_path'] == str(model_dir)


def test_model_manager_starts_when_completed_replacement_backup_cannot_be_removed(tmp_path):
    model_dir = tmp_path / 'testhash123'
    model_dir.mkdir()
    (model_dir / 'new.keras').write_bytes(b'new model')
    backup_dir = tmp_path / '@backup-testhash123'
    backup_dir.mkdir()
    (backup_dir / 'old.keras').write_bytes(b'old model')

    with patch.object(ModelManager, '_remove_path', side_effect=OSError('simulated persistent cleanup failure')):
        manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    assert (model_dir / 'new.keras').read_bytes() == b'new model'
    assert (backup_dir / 'old.keras').read_bytes() == b'old model'
    assert manager.metadata_store['testhash123']['file_path'] == str(model_dir)


@patch('src.core.model_manager.tf.keras.models.load_model')
def test_upload_model_replaces_existing_hash_and_invalidates_cached_model(mock_load_model, tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'old.keras').write_bytes(b'old model')

    old_model = MagicMock()
    old_model.predict.return_value = np.array([[0.1, 0.9]])
    manager.model_cache[model_hash] = old_model
    manager.metadata_store[model_hash] = {
        'file_path': str(model_dir),
        'used': '2024-04-27T12:00:00',
    }

    uploaded_model = _UploadFile(create_test_model_zip_bytes())

    replacement_model = MagicMock()
    replacement_model.predict.return_value = np.array([[0.8, 0.2]])
    mock_load_model.return_value = replacement_model

    response_message, response_status = manager.upload_model(uploaded_model, model_hash)

    assert response_status == 200
    assert response_message == 'Model uploaded successfully'
    assert not (model_dir / 'old.keras').exists()
    assert (model_dir / 'replacement-model.keras').exists()
    assert set(path.name for path in model_dir.glob('*.keras')) == {'replacement-model.keras'}
    assert model_hash not in manager.model_cache

    prediction, status = manager.predict(model_hash, np.array([[1.0, 2.0]]))
    assert status == 200
    assert prediction.tolist() == [[0.8, 0.2]]
    old_model.predict.assert_not_called()
    assert mock_load_model.call_count == 1


def test_upload_model_restores_existing_model_when_staged_install_fails(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'old.keras').write_bytes(b'old model')

    old_model = MagicMock()
    old_metadata = {
        'file_path': str(model_dir),
        'used': '2024-04-27T12:00:00',
    }
    manager.model_cache[model_hash] = old_model
    manager.metadata_store[model_hash] = old_metadata
    real_replace = os.replace

    def fail_staged_install(source, destination):
        if Path(source).name.startswith(f'@upload-{model_hash}-') and Path(destination) == model_dir:
            raise OSError('simulated staged install failure')
        return real_replace(source, destination)

    with patch('src.core.model_manager.os.replace', side_effect=fail_staged_install):
        with pytest.raises(OSError, match='simulated staged install failure'):
            manager.upload_model(_UploadFile(create_test_model_zip_bytes()), model_hash)

    assert (model_dir / 'old.keras').read_bytes() == b'old model'
    assert not (tmp_path / f'@backup-{model_hash}').exists()
    assert manager.metadata_store[model_hash] == old_metadata
    assert manager.model_cache[model_hash] is old_model


def test_upload_model_preserves_existing_model_when_backup_rename_fails(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'old.keras').write_bytes(b'old model')
    old_model = MagicMock()
    old_metadata = {
        'file_path': str(model_dir),
        'used': '2024-04-27T12:00:00',
    }
    manager.model_cache[model_hash] = old_model
    manager.metadata_store[model_hash] = old_metadata
    real_replace = os.replace

    def fail_backup_rename(source, destination):
        if Path(source) == model_dir:
            raise OSError('simulated backup rename failure')
        return real_replace(source, destination)

    with patch('src.core.model_manager.os.replace', side_effect=fail_backup_rename):
        with pytest.raises(OSError, match='simulated backup rename failure'):
            manager.upload_model(_UploadFile(create_test_model_zip_bytes()), model_hash)

    assert (model_dir / 'old.keras').read_bytes() == b'old model'
    assert not (tmp_path / f'@backup-{model_hash}').exists()
    assert manager.metadata_store[model_hash] == old_metadata
    assert manager.model_cache[model_hash] is old_model


def test_upload_model_failed_install_for_new_hash_leaves_no_model_state(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    real_replace = os.replace

    def fail_staged_install(source, destination):
        if Path(source).name.startswith(f'@upload-{model_hash}-') and Path(destination) == model_dir:
            raise OSError('simulated new model install failure')
        return real_replace(source, destination)

    with patch('src.core.model_manager.os.replace', side_effect=fail_staged_install):
        with pytest.raises(OSError, match='simulated new model install failure'):
            manager.upload_model(_UploadFile(create_test_model_zip_bytes()), model_hash)

    assert not model_dir.exists()
    assert not (tmp_path / f'@backup-{model_hash}').exists()
    assert model_hash not in manager.metadata_store
    assert model_hash not in manager.model_cache


def test_upload_model_leaves_recoverable_backup_when_immediate_rollback_fails(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'old.keras').write_bytes(b'old model')
    backup_dir = tmp_path / f'@backup-{model_hash}'
    real_replace = os.replace

    def fail_install_and_rollback(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(f'@upload-{model_hash}-') and destination_path == model_dir:
            raise OSError('simulated staged install failure')
        if source_path == backup_dir and destination_path == model_dir:
            raise OSError('simulated immediate rollback failure')
        return real_replace(source, destination)

    with patch('src.core.model_manager.os.replace', side_effect=fail_install_and_rollback):
        with pytest.raises(OSError, match='Failed to install model testhash123 and restore its previous version'):
            manager.upload_model(_UploadFile(create_test_model_zip_bytes()), model_hash)

    assert not model_dir.exists()
    assert (backup_dir / 'old.keras').read_bytes() == b'old model'

    restarted_manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    assert (model_dir / 'old.keras').read_bytes() == b'old model'
    assert not backup_dir.exists()
    assert restarted_manager.metadata_store[model_hash]['file_path'] == str(model_dir)


def test_upload_model_keeps_new_model_when_backup_cleanup_is_deferred(tmp_path):
    manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)
    model_hash = 'testhash123'
    model_dir = tmp_path / model_hash
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / 'old.keras').write_bytes(b'old model')
    backup_dir = tmp_path / f'@backup-{model_hash}'
    real_remove_path = manager._remove_path

    def defer_backup_cleanup(path):
        if Path(path) == backup_dir:
            raise OSError('simulated backup cleanup failure')
        return real_remove_path(path)

    with patch.object(manager, '_remove_path', side_effect=defer_backup_cleanup):
        response_message, response_status = manager.upload_model(
            _UploadFile(create_test_model_zip_bytes()),
            model_hash,
        )

    assert response_status == 200
    assert response_message == 'Model uploaded successfully'
    assert (model_dir / 'replacement-model.keras').read_bytes() == b'replacement content'
    assert (backup_dir / 'old.keras').read_bytes() == b'old model'

    restarted_manager = ModelManager(str(tmp_path), cleanup_interval_hours=0)

    assert not backup_dir.exists()
    assert (model_dir / 'replacement-model.keras').read_bytes() == b'replacement content'
    assert restarted_manager.metadata_store[model_hash]['file_path'] == str(model_dir)
