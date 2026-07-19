import io

import pytest

from src.common.errors import ErrorCode
from src.api.error_handlers import ERROR_SPECS


def test_every_error_code_has_one_public_contract():
    assert set(ERROR_SPECS) == set(ErrorCode)


def test_unexpected_error_uses_stable_internal_contract(client, monkeypatch, assert_error_response):
    def fail_lookup(model_hash):
        raise RuntimeError('sensitive storage detail')

    monkeypatch.setattr(client.application.model_manager, 'get_model_info', fail_lookup)

    response = client.get('/get_model?hash=testhash123')

    assert_error_response(response, 500, ErrorCode.INTERNAL_ERROR)
    assert 'sensitive storage detail' not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    ('path', 'status_code', 'error_code'),
    [
        ('/missing', 404, ErrorCode.ROUTE_NOT_FOUND),
        ('/predict', 405, ErrorCode.METHOD_NOT_ALLOWED),
    ],
)
def test_http_routing_errors_use_stable_contract(
    client,
    get_metric_value,
    assert_error_response,
    path,
    status_code,
    error_code,
):
    before = get_metric_value('ml_api_errors', {'type': error_code.value}) or 0

    response = client.get(path)

    assert_error_response(response, status_code, error_code)
    after = get_metric_value('ml_api_errors', {'type': error_code.value})
    assert after == before + 1


def test_method_not_allowed_preserves_allow_header(client):
    response = client.get('/predict')

    assert response.status_code == 405
    assert 'POST' in response.headers['Allow']


def test_unmapped_http_errors_use_stable_fallback_contract(
    client,
    get_metric_value,
    assert_error_response,
):
    client.application.config['MAX_FORM_PARTS'] = 1
    before = get_metric_value('ml_api_errors', {'type': ErrorCode.HTTP_ERROR.value}) or 0

    response = client.post(
        '/upload_model?hash=testhash123',
        data={
            'extra': 'value',
            'model_file': (io.BytesIO(b'zip'), 'model.zip'),
        },
        content_type='multipart/form-data',
    )

    assert_error_response(
        response,
        413,
        ErrorCode.HTTP_ERROR,
        {'status': 413},
    )
    after = get_metric_value('ml_api_errors', {'type': ErrorCode.HTTP_ERROR.value})
    assert after == before + 1


def test_oversized_json_is_rejected_before_prediction_parsing(
    client,
    assert_error_response,
):
    client.application.config['MAX_CONTENT_LENGTH'] = 32

    response = client.post(
        '/predict?hash=testhash123',
        data=b'[' + (b'0,' * 32) + b'0]',
        content_type='application/json',
    )

    assert_error_response(
        response,
        413,
        ErrorCode.HTTP_ERROR,
        {'status': 413},
    )
