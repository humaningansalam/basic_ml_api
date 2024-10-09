import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from zipfile import ZipFile
import numpy as np
import keras
from collections import OrderedDict

from flask import Flask, request, jsonify

from prometheus_flask_exporter import PrometheusMetrics

from monitor import ResourceMonitor
from comm_.prometheus_metric import get_metrics
import comm_.tool_util as tool_util

#flask
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# 메타데이터 저장소 및 모델 캐시 초기화
metadata_store = {}
model_cache = OrderedDict()

# 모델 저장 경로 및 캐시 사이즈 설정
model_store_path = "../data/model_"
max_cache_size = 10  # 최대 캐시할 모델 개수

# Prometheus 메트릭스 인스턴스 가져오기
prometheus_metrics = get_metrics()

def load_model_to_cache(model_hash):
    """
    모델을 캐시에 로드하는 함수
    """
    try:
        if model_hash in model_cache:
            # 모델이 이미 캐시에 있는 경우, 순서 갱신
            model_cache.move_to_end(model_hash)
        else:
            if len(model_cache) >= max_cache_size:
                # 캐시가 꽉 찼을 경우, 가장 오래된 항목 제거
                oldest_key, _ = model_cache.popitem(last=False)
                logging.info(f"Removing oldest model from cache: {oldest_key}")

            # 모델 로드
            model_file_path = metadata_store[model_hash]['file_path']
            model = keras.models.load_model(model_file_path)
            model_cache[model_hash] = model
            logging.info(f"Model {model_hash} loaded into cache successfully.")
    except Exception as e:
        logging.error(f"Unexpected error occurred while loading model to cache: {str(e)}")


@app.route('/upload_model', methods=['POST'])
def upload_model():
    """
    모델을 받아오는 함수
    """

    model_file = request.files.get('model_file')
    model_hash = request.args.get('hash')

    # 입력 검사
    if not model_file or not model_hash:
        response_message = 'Model file and hash are required'
        logging.error(response_message)
        return jsonify({'error': response_message}), 400

    # 해시값을 이름으로 가진 폴더 생성
    model_folder_path = os.path.join(model_store_path, model_hash)
    os.makedirs(model_folder_path, exist_ok=True)
    
    try:
        # 모델 파일 저장
        model_file_path = os.path.join(model_folder_path, model_file.filename)
        model_file.save(model_file_path)
        logging.debug(f'Model file saved at {model_file_path}')

        # .zip 파일 압축 해제
        with ZipFile(model_file_path, 'r') as zipObj:
            # 전체 Zip 파일의 압축을 해제
            zipObj.extractall(model_folder_path)
            logging.debug(f'Extracted zip file to {model_folder_path}')

        # 압축 해제 후 원본 .zip 파일 삭제
        os.remove(model_file_path)
        logging.debug('Original zip file removed')

        # 메타데이터 저장
        metadata_store[model_hash] = {'file_path': model_folder_path,
                                       'used': tool_util.get_kr_time()}
        logging.debug('Metadata saved')

        response_message = 'File uploaded and processed successfully'
        logging.info(response_message)
        return jsonify({'message': response_message}), 200

    except Exception as e:
        prometheus_metrics.increment_error_count('upload_model_error')
        response_message = f'An error occurred: {str(e)}'
        logging.error(response_message)
        
        # 예외 발생 시, 생성된 폴더 삭제
        if os.path.exists(model_folder_path):
            shutil.rmtree(model_folder_path)
            logging.info(f'Removed folder {model_folder_path} due to an error')

        return jsonify({'error': response_message}), 500

@app.route('/get_model', methods=['GET'])
def get_model():
    """
    모델 존재 확인 함수
    """
    # 해시값 받기
    model_hash = request.args.get('hash')

    # 입력 검사
    if not model_hash:
        response_message = 'Model hash are required'
        logging.error(response_message)
        return jsonify({'error': response_message}), 400
    
    # 메타데이터 조회
    if model_hash not in metadata_store:
        response_message = 'No such model'
        logging.error(response_message)
        return jsonify({'message':response_message}), 404

    metadata = metadata_store[model_hash]
    return jsonify({'message': metadata}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    모델 결과 반환 함수
    """
    try:
        # 데이터 및 모델 해시값 받기
        data = request.get_json()
        model_hash = request.args.get('hash')

        # 입력 검사
        if not data or not model_hash:
            response_message = 'data or Model hash are required'
            logging.error(response_message)
            return jsonify({'error': response_message}), 400
    
        # 캐시에서 모델 확인
        if model_hash not in model_cache:
            # 캐시에 없을 경우, 캐시에 로드
            load_model_to_cache(model_hash)
        model = model_cache[model_hash]

        # 모델 사용시간 업데이트
        metadata_store[model_hash]['used'] = tool_util.get_kr_time()

        # 데이터를 numpy 배열로 변환 및 예측
        array = np.array(data)
        prediction = model.predict(array)

        # 결과 완료 카운트
        prometheus_metrics.increment_predictions_completed()

        # 결과 반환
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        prometheus_metrics.increment_error_count('predict')
        response_message = f'An error occurred: {str(e)}'
        logging.error(response_message)
        return jsonify({'error': response_message}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    flask health check 
    """
    return "Healthy"

def load_metadata_store():
    """
    flask health check 
    """
    # 모든 모델 폴더 순회
    for model_hash in os.listdir(model_store_path):
        model_folder_path = os.path.join(model_store_path, model_hash)
        if os.path.isdir(model_folder_path):

            # metadata_store에 메타데이터 추가
            metadata_store[model_hash] = {
                'file_path': model_folder_path,
                'used': tool_util.get_kr_time()
            }

def del_oldmodel():
    """
    del old model
    """
    to_remove = []  # 제거할 항목의 키를 저장할 리스트
    for model_hash, metadata in metadata_store.items():
        if metadata['used'] < tool_util.one_week_ago():
            to_remove.append(model_hash)
    # 제거할 항목을 metadata_store에서 제거
    for model_hash in to_remove:
        to_remove_path = metadata_store[model_hash]['file_path']
        if os.path.exists(to_remove_path):
            shutil.rmtree(to_remove_path)
        del metadata_store[model_hash]

def sched_del_oldmodel():
    """
    sched old model
    AM 5 del_oldmodel 실행
    """
    threading.Timer(tool_util.delay_h(5), sched_del_oldmodel).start()

    del_oldmodel()
    
if __name__ == '__main__':

    log_level = os.getenv('LOG_LEVEL', 'INFO')
    tool_util.set_logging(log_level)
    tool_util.set_folder()

    load_metadata_store()

    with ThreadPoolExecutor() as executor:
        executor.submit(sched_del_oldmodel())

    with ThreadPoolExecutor() as executor:
        resource_monitor = ResourceMonitor()
        executor.submit(resource_monitor.start_monitor)

    app.run(host='0.0.0.0', port=5000)