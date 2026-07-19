from datetime import datetime, timezone

from src.common.errors import ErrorCode
from src.core.model_types import ModelMetadata


def test_get_model_success(client):
    used_at = datetime(2024, 4, 27, 12, 0, tzinfo=timezone.utc)
    client.application.model_manager.metadata_store['testhash123'] = ModelMetadata(
        file_path='../data/model_/testhash123',
        used=used_at,
    )

    response = client.get('/get_model?hash=testhash123')

    assert response.status_code == 200
    assert response.json == {
        'data': {
            'model_hash': 'testhash123',
            'file_path': '../data/model_/testhash123',
            'used': used_at.isoformat(),
        }
    }


def test_get_model_missing_hash(client, get_metric_value, assert_error_response):
    response = client.get('/get_model')

    assert_error_response(response, 400, ErrorCode.MODEL_HASH_REQUIRED)
    assert get_metric_value('ml_api_errors', {'type': ErrorCode.MODEL_HASH_REQUIRED.value}) == 1


def test_get_model_not_found(client, get_metric_value, assert_error_response):
    before = get_metric_value('ml_api_errors', {'type': ErrorCode.MODEL_NOT_FOUND.value}) or 0
    response = client.get('/get_model?hash=nonexistent')

    assert_error_response(
        response,
        404,
        ErrorCode.MODEL_NOT_FOUND,
        {'model_hash': 'nonexistent'},
    )
    after = get_metric_value('ml_api_errors', {'type': ErrorCode.MODEL_NOT_FOUND.value})
    assert after == before + 1
