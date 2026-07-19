import pytest
from prometheus_client import REGISTRY

from src.common.errors import ErrorCode
from src.config import Config
from src.main import create_app

@pytest.fixture
def app(tmp_path):
    """테스트용 Flask 애플리케이션을 반환하는 fixture"""
    class TestConfig(Config):
        TESTING = True
        START_BACKGROUND_MONITORING = False
        MODEL_STORE_PATH = str(tmp_path / 'models')
        MODEL_CLEANUP_INTERVAL = 0

    app = create_app(TestConfig)
    return app

@pytest.fixture
def client(app):
    """Flask 테스트 클라이언트를 반환하는 fixture"""
    return app.test_client()

@pytest.fixture
def get_metric_value():
    """프로메테우스 메트릭 값을 조회하는 helper fixture"""
    def _get_metric_value(metric_name, labels=None):
        for metric in REGISTRY.collect():
            if metric.name == metric_name:
                for sample in metric.samples:
                    if labels is None or sample.labels == labels:
                        return sample.value
        return None
    return _get_metric_value


@pytest.fixture
def assert_error_response():
    def _assert_error_response(response, status_code, error_code: ErrorCode, details=None):
        assert response.status_code == status_code
        assert response.content_type == 'application/json'
        assert response.json['error']['code'] == error_code.value
        assert isinstance(response.json['error']['message'], str)
        assert response.json['error']['details'] == (details or {})

    return _assert_error_response
