import numpy as np
from unittest.mock import patch, MagicMock
from collections import OrderedDict

@patch('myapp.src.model_manager.os.path.exists')
@patch('keras.models.load_model')
@patch('myapp.common.prometheus_metric.get_metrics')
def test_predict_success(mock_get_metrics, mock_load_model, mock_exists, client):
    """예측 성공 테스트"""
    # 설정
    mock_exists.return_value = True  # os.path.exists가 항상 True를 반환하도록 Mock 처리
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.8, 0.2]])
    mock_load_model.return_value = mock_model

    mock_metrics = MagicMock()
    mock_get_metrics.return_value = mock_metrics

    test_metadata = {
        'file_path': '../data/model_/testhash123',
        'used': '2024-04-27T12:00:00'
    }
    client.application.model_manager.metadata_store['testhash123'] = test_metadata
    client.application.model_manager.model_cache = OrderedDict()

    # 테스트 실행
    response = client.post('/predict?hash=testhash123', json=[[0.5, 0.5]])

    # 검증
    assert response.status_code == 200
    assert response.json['prediction'] == [[0.8, 0.2]]
    mock_metrics.increment_predictions_completed.assert_called_once()
    mock_metrics.increment_cache_misses.assert_called_once()

@patch('myapp.common.prometheus_metric.get_metrics')
def test_predict_missing_data(mock_get_metrics, client):
    """데이터 누락 테스트"""
    mock_metrics = MagicMock()
    mock_get_metrics.return_value = mock_metrics

    # 데이터 없이 해시만 보내는 케이스
    response = client.post('/predict?hash=testhash123', data='')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Data is required'

    mock_metrics.increment_error_count.assert_called_with('predict_missing_data')

@patch('myapp.common.prometheus_metric.get_metrics')
def test_predict_model_not_found(mock_get_metrics, client):
    """존재하지 않는 모델로 예측 시도 테스트"""
    mock_metrics = MagicMock()
    mock_get_metrics.return_value = mock_metrics

    response = client.post('/predict?hash=nonexistent', 
                         json=[[0.5, 0.5]])
    
    assert response.status_code == 404
    assert 'error' in response.json

    mock_metrics.increment_error_count.assert_called_with('predict_model_not_found')
