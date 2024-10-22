from unittest.mock import patch

@patch('myapp.src.main.tool_util.get_kr_time', return_value='2024-04-27T12:00:00')
def test_get_model_success(mock_time, client):
    """모델 조회 성공 테스트"""
    test_metadata = {
        'file_path': '../data/model_/testhash123',
        'used': '2024-04-27T12:00:00'
    }
    client.application.model_manager.metadata_store['testhash123'] = test_metadata
    
    response = client.get('/get_model?hash=testhash123')
    
    assert response.status_code == 200
    assert response.json['message']['file_path'] == test_metadata['file_path']

def test_get_model_missing_hash(client, get_metric_value):
    """해시 파라미터 누락 테스트"""
    response = client.get('/get_model')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Model hash is required'
    
    error_count = get_metric_value('errors', {'type': 'get_model_missing_hash'})
    assert error_count == 1

def test_get_model_not_found(client, get_metric_value):
    """존재하지 않는 모델 조회 테스트"""
    response = client.get('/get_model?hash=nonexistent')
    
    assert response.status_code == 404
    assert response.json['message'] == 'No such model'
    
    error_count = get_metric_value('errors', {'type': 'get_model_not_found'})
    assert error_count == 1