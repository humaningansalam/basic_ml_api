# tests/test_predict.py
from typing import OrderedDict
import numpy as np
from unittest.mock import patch, MagicMock

@patch('keras.models.load_model')
@patch('myapp.common.tool_util.get_kr_time', return_value='2024-04-27T12:00:00')
def test_predict_success(mock_time, mock_keras_load, client, get_counter_value):
    # Keras 모델 모킹
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1, 2, 3])
    mock_keras_load.return_value = mock_model  # keras.models.load_model이 모킹된 모델을 반환하도록 설정

    # 메타데이터 설정
    client.application.metadata_store['testhash123'] = {
        'file_path': '../data/model_testhash123',
        'used': '2024-04-27T12:00:00'
    }

    # 모델 캐시 초기화
    client.application.model_cache = OrderedDict()

    # 예측 요청 보내기
    data = [0.1, 0.2, 0.3]
    response = client.post('/predict?hash=testhash123', json=data)

    # 응답 확인
    assert response.status_code == 200
    assert response.json['prediction'] == [1, 2, 3]

    # keras.models.load_model이 올바르게 호출되었는지 확인
    mock_keras_load.assert_called_once_with('../data/model_testhash123')

    # 예측 카운트가 증가했는지 확인
    counter_value = get_counter_value('predictions_completed', {})
    assert counter_value == 1

@patch('myapp.common.tool_util.get_kr_time')
def test_predict_missing_data(mock_time, client, get_counter_value):
    response = client.post('/predict', json={}, content_type='application/json')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Data and Model hash are required'
    
    # 에러 카운트가 증가했는지 확인
    counter_value = get_counter_value('errors', {'type':'predict_missing_data'})
    assert counter_value == 1

def test_predict_model_load_failed(client, get_counter_value):
    # 사전 메타데이터 설정
    client.application.metadata_store['testhash123'] = {
        'file_path': '../data/non_existent_model',
        'used': '2024-04-27T12:00:00'
    }

    client.post('/reset_cache')

    data = [0.1, 0.2, 0.3]
    response = client.post('/predict?hash=testhash123', json=data)

    assert response.status_code == 500
    assert response.json['error'] == 'Model file not found' 

    counter_value = get_counter_value('errors', {'type': 'predict_model_load_failed'})
    assert counter_value == 1