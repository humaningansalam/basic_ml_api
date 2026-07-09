import pytest
import io
import zipfile
import numpy as np
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
    mock_zip_instance.namelist.return_value = ['model.keras']
    mock_zip_instance.getinfo.return_value = MagicMock(file_size=100)

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
    mock_zip.namelist.return_value = ['model.txt']
    mock_zip.getinfo.return_value = MagicMock(file_size=100)
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
    mock_zip.namelist.return_value = ['../escape.txt', 'safe.keras']
    mock_zip.getinfo.return_value = MagicMock(file_size=100)
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
