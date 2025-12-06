import io
import zipfile
from unittest.mock import patch, MagicMock

def create_test_model_zip():
    """테스트용 모델 ZIP 파일 생성 (keras 파일 포함)"""
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        zf.writestr('model.keras', b'dummy content')
    memory_file.seek(0)
    return memory_file

@patch('werkzeug.datastructures.FileStorage.save')
@patch('src.core.model_manager.ZipFile')
@patch('src.core.model_manager.os.remove')
@patch('src.core.model_manager.os.makedirs')
def test_upload_model_success(mock_makedirs, mock_remove, mock_zipfile, mock_save, client):
    """모델 업로드 성공 테스트"""
    # ZipFile 동작 모킹
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ['model.keras'] 

    test_zip = create_test_model_zip()

    # 테스트 실행
    response = client.post('/upload_model?hash=testhash123',
                         data={'model_file': (test_zip, 'model.zip')},
                         content_type='multipart/form-data')
    
    mock_save.assert_called_once()
    
    assert response.status_code == 200
    assert response.json['message'] == 'Model uploaded successfully'

def test_upload_model_missing_data(client, get_metric_value):
    """필수 데이터 누락 테스트"""
    response = client.post('/upload_model')
    
    assert response.status_code == 400
    assert 'Missing data' in response.json['error']
    
    counter_value = get_metric_value('errors', {'type': 'upload_model_missing_data'})
    assert counter_value == 1