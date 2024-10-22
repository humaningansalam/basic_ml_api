import pytest
from prometheus_client import REGISTRY

from myapp.src.main import create_app

@pytest.fixture
def app():
    """테스트용 Flask 애플리케이션을 반환하는 fixture"""
    return create_app() 

@pytest.fixture
def client(app):
    """Flask 테스트 클라이언트를 반환하는 fixture"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Flask CLI runner를 반환하는 fixture"""
    return app.test_cli_runner()

@pytest.fixture
def get_counter_value():
    def _get_counter_value(counter_name, label_values):
        for metric in REGISTRY.collect():
            if metric.name == counter_name:
                for sample in metric.samples:
                    if sample.labels == label_values:
                        return sample.value
        return None
    return _get_counter_value