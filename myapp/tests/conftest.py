import os
import pytest
from prometheus_client import REGISTRY

from myapp.src.main import create_app

@pytest.fixture
def app():
    """테스트용 Flask 애플리케이션을 반환하는 fixture"""
    app = create_app()
    app.testing = True # 테스트 모드로 설정하여 백그라운드 작업 비활성화
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