from flask import Flask, request, jsonify
import os
import keras
import shutil
from zipfile import ZipFile
import numpy as np
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge, Counter

import resource_monitor as resource_monitor

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# 메타데이터 저장소
metadata_store = {}
model_cache = {}

# 모델 저장 경로
model_store_path = "./model_"
max_cache_size = 10  # 최대 캐시할 모델 개수


# 프로메테우스 지표 정의
model_cache_usage = Gauge('model_cache_usage', 'Number of models currently cached')
errors_count = Counter('errors', 'Number of errors')
predictions_completed = Counter('predictions_completed', 'Number of completed predictions')

def load_model_to_cache(model_hash):
    """모델을 캐시에 로드하는 함수"""
    if len(model_cache) >= max_cache_size:
        # 캐시가 꽉 찼을 경우, 가장 오래된 항목 제거
        oldest_key = next(iter(model_cache))
        del model_cache[oldest_key]
    model_file_path = metadata_store[model_hash]['file_path']
    model = keras.models.load_model(model_file_path)
    model_cache[model_hash] = model
    # 프로메테우스 모델 캐시 사용량
    model_cache_usage.set(len(model_cache))


@app.route('/upload_model', methods=['POST'])
def upload_model():
    # 모델 파일 받기
    model_file = request.files['model_file']
    # 해시값 받기
    model_hash = request.args.get('hash')

    # 해시값을 이름으로 가진 폴더 생성
    model_folder_path = os.path.join(model_store_path, model_hash)
    os.makedirs(model_folder_path, exist_ok=True)
    
    try:
        # 모델 파일 저장
        model_file_path = os.path.join(model_folder_path, model_file.filename)
        model_file.save(model_file_path)

        # 파일이 .zip 형태인 경우 압축 해제
        if model_file.filename.endswith('.zip'):
            with ZipFile(model_file_path, 'r') as zipObj:
                # 전체 Zip 파일의 압축을 해제
                zipObj.extractall(model_folder_path)

        # 압축 해제 후 원본 .zip 파일 삭제
        if os.path.exists(model_file_path):
            os.remove(model_file_path)

        # 메타데이터 저장
        metadata_store[model_hash] = {'file_path': model_folder_path}

        return 'File uploaded, unzipped and original zip file removed successfully', 200
    except Exception as e:
        errors_count.labels(type='predict').inc()
        # 예외 발생 시, 생성된 폴더 삭제
        if os.path.exists(model_folder_path):
            shutil.rmtree(model_folder_path)
        return f'An error occurred: {str(e)}', 500
    

@app.route('/get_model', methods=['GET'])
def get_model():
    # 해시값 받기
    model_hash = request.args.get('hash')

    # 메타데이터 조회
    if model_hash not in metadata_store:
        return 'No such model', 404

    metadata = metadata_store[model_hash]
    return jsonify(metadata), 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 데이터 및 모델 해시값 받기
        data = request.get_json()
        model_hash = request.args.get('hash')

        # 캐시에서 모델 확인
        if model_hash not in model_cache:
            # 캐시에 없을 경우, 캐시에 로드
            load_model_to_cache(model_hash)
        
        model = model_cache[model_hash]

        # 데이터를 numpy 배열로 변환 및 예측
        array = np.array(data)
        prediction = model.predict(array)

        # 결과 완료 카운트
        predictions_completed.inc()
        # 결과 반환
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        errors_count.labels(type='predict').inc()
        return f'An error occurred: {str(e)}', 500

@app.route('/health', methods=['GET'])
def health():
    return "Healthy"

if __name__ == '__main__':
    resource_monitor.threading_app_resource()
    app.run(host='0.0.0.0', port=5000)