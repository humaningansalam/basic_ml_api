# tests/test_get_model.py
from unittest.mock import patch

@patch('myapp.src.main.tool_util.get_kr_time', return_value='2024-04-27T12:00:00')
def test_get_model_success(mock_time, client):
    # 사전 메타데이터 설정
    client.application.metadata_store['testhash123'] = {
        'file_path': '../data/model_testhash123',
        'used': '2024-04-27T12:00:00'
    }
    
    response = client.get('/get_model?hash=testhash123')
    
    assert response.status_code == 200
    assert response.json['message'] == {
        'file_path': '../data/model_testhash123',
        'used': '2024-04-27T12:00:00'
    }

def test_get_model_not_found(client):
    response = client.get('/get_model?hash=nonexistenthash')
    
    assert response.status_code == 404
    assert response.json['message'] == 'No such model'
    
    # 에러 카운트가 증가했는지 확인
    assert client.application.metrics.errors_count.labels(type='get_model_not_found').value == 1

def test_get_model_missing_hash(client):
    response = client.get('/get_model')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Model hash is required'
    
    # 에러 카운트가 증가했는지 확인
    assert client.application.metrics.errors_count.labels(type='get_model_missing_hash').value == 1
