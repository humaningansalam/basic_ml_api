import io
import zipfile
from unittest.mock import patch, MagicMock

def create_test_model_zip():
    """테스트용 모델 ZIP 파일 생성"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('saved_model.pb', b'dummy model content')
        zf.writestr('variables/variables.data-00000-of-00001', b'dummy variables')
        zf.writestr('variables/variables.index', b'dummy index')
    memory_file.seek(0)
    return memory_file

@patch('keras.models.load_model')
@patch('myapp.common.prometheus_metric.get_metrics')
def test_upload_model_success(mock_get_metrics, mock_load_model, client):
    """모델 업로드 성공 테스트"""
    # 테스트 설정
    mock_load_model.return_value = MagicMock()
    mock_metrics = MagicMock()
    mock_get_metrics.return_value = mock_metrics

    test_zip = create_test_model_zip()
    
    # 테스트 실행
    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (test_zip, 'model.zip')},
                         content_type='multipart/form-data')
    
    # 검증
    assert response.status_code == 200
    assert response.json['message'] == 'File uploaded and processed successfully'
    assert 'testhash123' in client.application.model_manager.metadata_store

def test_upload_model_missing_data(client, get_metric_value):
    """필수 데이터 누락 테스트"""
    mock_metrics = MagicMock()

    response = client.post('/upload_model')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Model file and hash are required'
    
    counter_value = get_metric_value('errors', {'type': 'upload_model_missing_data'})
    assert counter_value == 1
