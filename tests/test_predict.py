from collections import OrderedDict
from datetime import datetime, timezone
import gc
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.errors import ErrorCode
from src.core.model_types import ModelMetadata


def _register_model_file(manager, model_hash='testhash123'):
    model_dir = Path(manager.store_path) / model_hash
    model_dir.mkdir()
    (model_dir / 'model.keras').write_bytes(b'model')
    manager.metadata_store[model_hash] = ModelMetadata(
        file_path=str(model_dir),
        used=datetime(2024, 4, 27, 12, 0, tzinfo=timezone.utc),
    )


@patch('tensorflow.keras.models.load_model')
def test_predict_success(mock_load_model, client, get_metric_value):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2]])
    mock_load_model.return_value = mock_model

    manager = client.application.model_manager
    _register_model_file(manager)
    manager.model_cache = OrderedDict()

    response = client.post('/predict?hash=testhash123', json=[[0.5, 0.5]])

    assert response.status_code == 200
    assert response.json == {
        'data': {
            'model_hash': 'testhash123',
            'prediction': [[0.8, 0.2]],
        }
    }
    assert get_metric_value('predictions_completed') == 1


@pytest.mark.parametrize(
    ('model_output', 'expected_prediction'),
    [
        (
            [np.array([[8.0]]), np.array([[12.0]])],
            [[[8.0]], [[12.0]]],
        ),
        (
            {
                'double': np.array([[8.0]]),
                'triple': np.array([[12.0]]),
            },
            {
                'double': [[8.0]],
                'triple': [[12.0]],
            },
        ),
    ],
)
@patch('tensorflow.keras.models.load_model')
def test_predict_serializes_nested_model_outputs(
    mock_load_model,
    client,
    model_output,
    expected_prediction,
):
    mock_model = MagicMock()
    mock_model.predict.return_value = model_output
    mock_load_model.return_value = mock_model
    _register_model_file(client.application.model_manager)

    response = client.post('/predict?hash=testhash123', json=[[4.0]])

    assert response.status_code == 200
    assert response.json == {
        'data': {
            'model_hash': 'testhash123',
            'prediction': expected_prediction,
        }
    }


def test_predict_missing_hash(client, assert_error_response):
    response = client.post('/predict', json=[[0.5, 0.5]])
    assert_error_response(response, 400, ErrorCode.MODEL_HASH_REQUIRED)


def test_predict_missing_json_body(client, assert_error_response):
    response = client.post('/predict?hash=testhash123')
    assert_error_response(response, 400, ErrorCode.PREDICTION_DATA_REQUIRED)


def test_predict_empty_json_body(client, assert_error_response):
    response = client.post(
        '/predict?hash=testhash123',
        data=b'',
        content_type='application/json',
    )
    assert_error_response(response, 400, ErrorCode.PREDICTION_DATA_REQUIRED)


def test_predict_malformed_json_body(client, assert_error_response):
    response = client.post(
        '/predict?hash=testhash123',
        data=b'{',
        content_type='application/json',
    )
    assert_error_response(response, 400, ErrorCode.MALFORMED_JSON)


@pytest.mark.parametrize('payload', [b'[NaN]', b'[Infinity]', b'[-Infinity]'])
def test_predict_non_standard_numeric_tokens_are_malformed_json(
    client,
    payload,
    assert_error_response,
):
    response = client.post(
        '/predict?hash=unknown-model-v1',
        data=payload,
        content_type='application/json',
    )

    assert_error_response(response, 400, ErrorCode.MALFORMED_JSON)


def test_predict_json_null_returns_non_array_error(client, assert_error_response):
    response = client.post(
        '/predict?hash=testhash123',
        data=b'null',
        content_type='application/json',
    )
    assert_error_response(response, 400, ErrorCode.PREDICTION_DATA_NOT_ARRAY)


def test_predict_empty_array_returns_client_error(client, assert_error_response):
    response = client.post('/predict?hash=testhash123', json=[])
    assert_error_response(response, 400, ErrorCode.PREDICTION_DATA_EMPTY)


@pytest.mark.parametrize('payload', [{'x': 1}, 'abc', 1, True])
def test_predict_non_array_json_returns_client_error(client, payload, assert_error_response):
    with patch.object(client.application.model_manager, 'predict') as mock_predict:
        response = client.post('/predict?hash=testhash123', json=payload)

    assert_error_response(response, 400, ErrorCode.PREDICTION_DATA_NOT_ARRAY)
    mock_predict.assert_not_called()


def test_predict_ragged_array_returns_client_error(client, assert_error_response):
    response = client.post('/predict?hash=testhash123', json=[[1], [1, 2]])
    assert_error_response(response, 400, ErrorCode.INVALID_PREDICTION_DATA)


def test_predict_excessive_nesting_returns_client_error(client, assert_error_response):
    depth = sys.getrecursionlimit() + 100
    payload = ('[' * depth + '0' + ']' * depth).encode()

    response = client.post(
        '/predict?hash=testhash123',
        data=payload,
        content_type='application/json',
    )

    assert_error_response(response, 400, ErrorCode.INVALID_PREDICTION_DATA)


def test_predict_unknown_hashes_do_not_retain_model_locks(client):
    manager = client.application.model_manager

    for index in range(100):
        response = client.post(f'/predict?hash=missing-{index:08d}', json=[[1.0]])
        assert response.status_code == 404

    gc.collect()
    assert len(manager._cleanup_locks) == 0
    assert manager.metadata_store == {}
    assert manager.model_cache == OrderedDict()


def test_predict_non_finite_output_uses_prediction_failure_contract(
    client,
    assert_error_response,
):
    manager = client.application.model_manager
    model = MagicMock()
    model.predict.return_value = np.array([[np.nan, np.inf, -np.inf]])
    manager.model_cache['testhash123'] = model
    manager.metadata_store['testhash123'] = ModelMetadata(
        file_path='../data/model_/testhash123',
        used=datetime(2024, 4, 27, 12, 0, tzinfo=timezone.utc),
    )

    response = client.post('/predict?hash=testhash123', json=[[1.0]])

    assert_error_response(
        response,
        500,
        ErrorCode.PREDICTION_FAILED,
        {'model_hash': 'testhash123'},
    )
    assert 'NaN' not in response.get_data(as_text=True)
    assert 'Infinity' not in response.get_data(as_text=True)


def test_predict_model_not_found(client, get_metric_value, assert_error_response):
    before = get_metric_value('ml_api_errors', {'type': ErrorCode.MODEL_NOT_FOUND.value}) or 0
    response = client.post('/predict?hash=nonexistent', json=[[0.5, 0.5]])

    assert_error_response(
        response,
        404,
        ErrorCode.MODEL_NOT_FOUND,
        {'model_hash': 'nonexistent'},
    )
    after = get_metric_value('ml_api_errors', {'type': ErrorCode.MODEL_NOT_FOUND.value})
    assert after == before + 1
