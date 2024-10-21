import pytest

from myapp.src.main import app as flask_app

@pytest.fixture
def app():
    """Flask 애플리케이션을 반환하는 fixture"""
    return flask_app

@pytest.fixture
def client(app):
    """Flask 테스트 클라이언트를 반환하는 fixture"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Flask CLI runner를 반환하는 fixture"""
    return app.test_cli_runner()