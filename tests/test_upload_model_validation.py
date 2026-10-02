import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


def model_archive(tmp_path, multiplier):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1, use_bias=False),
    ])
    model.layers[0].set_weights([np.array([[multiplier]], dtype=np.float32)])
    model_path = tmp_path / 'model.keras'
    model.save(model_path)
    return archive_with_model(model_path.read_bytes())


def archive_with_model(content):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as zip_file:
        zip_file.writestr('nested/model.keras', content)
    archive.seek(0)
    return archive


@pytest.fixture(params=['not-a-model', 'invalid-keras-config'])
def invalid_archive(request):
    if request.param == 'not-a-model':
        return archive_with_model(b'not a Keras model')

    keras_file = io.BytesIO()
    with zipfile.ZipFile(keras_file, 'w') as zip_file:
        zip_file.writestr('config.json', '{invalid json')
    return archive_with_model(keras_file.getvalue())


def upload(client, archive):
    return client.post(
        '/upload_model?hash=testhash123',
        data={'model_file': (archive, 'model.zip')},
        content_type='multipart/form-data',
    )


def assert_prediction(client, expected):
    response = client.post('/predict?hash=testhash123', json=[[3.0]])
    assert response.status_code == 200
    assert response.json['data']['prediction'] == [[expected]]


def assert_invalid_artifact(response):
    assert response.status_code == 400
    assert response.json['error']['code'] == 'invalid_model_artifact'
    assert response.json['error']['details'] == {'model_hash': 'testhash123'}


def test_invalid_model_does_not_create_model_state(client, invalid_archive):
    assert_invalid_artifact(upload(client, invalid_archive))

    manager = client.application.model_manager
    assert manager.metadata_store == {}
    assert manager.model_cache == {}
    assert list(Path(manager.store_path).iterdir()) == []


@pytest.mark.parametrize('cached', [False, True])
def test_invalid_replacement_preserves_working_model(client, tmp_path, invalid_archive, cached):
    assert upload(client, model_archive(tmp_path, 2.0)).status_code == 200
    manager = client.application.model_manager
    model_path = Path(manager.store_path) / 'testhash123' / 'nested' / 'model.keras'
    original_bytes = model_path.read_bytes()
    if cached:
        assert_prediction(client, 6.0)
    original_cache = manager.model_cache.copy()
    original_metadata = manager.metadata_store['testhash123']
    original_used = original_metadata.used

    assert_invalid_artifact(upload(client, invalid_archive))

    assert model_path.read_bytes() == original_bytes
    assert manager.metadata_store['testhash123'] is original_metadata
    assert original_metadata.used == original_used
    assert manager.model_cache == original_cache
    assert sorted(path.name for path in Path(manager.store_path).iterdir()) == ['testhash123']
    assert_prediction(client, 6.0)

    # Check the on-disk model too, even if prediction used the original cache entry.
    manager.model_cache.clear()
    assert_prediction(client, 6.0)


def test_valid_replacement_still_replaces_cached_model(client, tmp_path):
    first_upload = upload(client, model_archive(tmp_path, 2.0))
    assert first_upload.json == {'data': {'model_hash': 'testhash123', 'replaced': False}}
    assert_prediction(client, 6.0)

    replacement = upload(client, model_archive(tmp_path, 5.0))

    assert replacement.status_code == 200
    assert replacement.json == {'data': {'model_hash': 'testhash123', 'replaced': True}}
    assert 'testhash123' not in client.application.model_manager.model_cache
    assert_prediction(client, 15.0)
