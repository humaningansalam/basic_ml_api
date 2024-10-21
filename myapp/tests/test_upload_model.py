# tests/test_upload_model.py
import io
import zipfile
import os
from unittest.mock import patch

def create_test_zip():
    """테스트용 ZIP 파일 생성"""
    bytes_io = io.BytesIO()
    with zipfile.ZipFile(bytes_io, 'w') as zipf:
        # 간단한 텍스트 파일 추가
        zipf.writestr('model.h5', 'dummy model content')
    bytes_io.seek(0)
    return bytes_io

@patch('myapp.src.main.load_model_to_cache')
@patch('myapp.common.tool_util.get_kr_time', return_value='2024-04-27T12:00:00')
def test_upload_model_success(mock_time, mock_load_model, client):
    test_zip = create_test_zip()
    data = {
        'model_file': (test_zip, 'model.zip')
    }
    response = client.post('/upload_model?hash=testhash123', data=data, content_type='multipart/form-data')
    
    assert response.status_code == 200
    assert response.json['message'] == 'File uploaded and processed successfully'
    
    # 모델 캐시 로드 함수가 호출되었는지 확인
    mock_load_model.assert_called_once_with('testhash123')
    
    # 메타데이터 저장소에 데이터가 추가되었는지 확인
    assert 'testhash123' in client.application.metadata_store
    assert client.application.metadata_store['testhash123']['file_path'] == os.path.join("../data/model_", 'testhash123')
    assert client.application.metadata_store['testhash123']['used'] == '2024-04-27T12:00:00'

@patch('myapp.src.main.load_model_to_cache')
@patch('myapp.common.tool_util.get_kr_time')
def test_upload_model_missing_data(mock_time, mock_load_model, client):
    response = client.post('/upload_model', data={}, content_type='multipart/form-data')
    
    assert response.status_code == 400
    assert response.json['error'] == 'Model file and hash are required'
    
    # 에러 카운트가 증가했는지 확인
    assert client.application.metrics.errors_count.labels(type='upload_model_missing_data').value == 1
