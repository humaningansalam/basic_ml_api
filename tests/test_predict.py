import numpy as np
from unittest.mock import patch, MagicMock
from collections import OrderedDict

@patch('src.core.model_manager.os.walk')
@patch('src.core.model_manager.os.path.exists')
@patch('tensorflow.keras.models.load_model') 
def test_predict_success(mock_load_model, mock_exists, mock_walk, client, get_metric_value):
    """예측 성공 테스트"""
    # 설정
    mock_exists.return_value = True
    
    # os.walk 모킹: .keras 파일을 찾을 수 있도록 설정
    mock_walk.return_value = [('/fake/path', [], ['model.keras'])]

    # 모델 예측 결과 모킹
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2]])
    mock_load_model.return_value = mock_model

    # 테스트 데이터 주입
    test_metadata = {
        'file_path': '../data/model_/testhash123',
        'used': '2024-04-27T12:00:00'
    }
    manager = client.application.model_manager
    manager.metadata_store['testhash123'] = test_metadata
    manager.model_cache = OrderedDict()

    # 테스트 실행
    response = client.post('/predict?hash=testhash123', json=[[0.5, 0.5]])

    # 검증
    assert response.status_code == 200
    assert response.json['prediction'] == [[0.8, 0.2]]

    counter_value = get_metric_value('predictions_completed')
    assert counter_value == 1

def test_predict_missing_data(client, get_metric_value):
    """데이터 누락 테스트"""
    # 데이터 없이 해시만 보내는 케이스
    response = client.post('/predict?hash=testhash123', json={})
    
    assert response.status_code == 400
    assert 'Missing' in response.json['error']
    counter_value = get_metric_value('ml_api_errors', {'type': 'predict_missing_data'})
    assert counter_value == 1

def test_predict_model_not_found(client, get_metric_value):
    """존재하지 않는 모델로 예측 시도 테스트"""
    response = client.post('/predict?hash=nonexistent', 
                         json=[[0.5, 0.5]])
    
    assert response.status_code == 404
    assert response.json['error'] == 'Model not found'

    counter_value = get_metric_value('ml_api_errors', {'type': 'predict_model_not_found'})
    assert counter_value == 1