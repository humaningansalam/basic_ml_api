import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from zipfile import ZipFile
from collections import OrderedDict
import numpy as np
import keras

from flask import Flask, request, jsonify, Response
from prometheus_client import generate_latest

from myapp.src.monitor import ResourceMonitor
from myapp.common.prometheus_metric import get_metrics
import myapp.common.tool_util as tool_util

# Flask 설정
app = Flask(__name__)

# Prometheus 메트릭스 초기화
metrics = get_metrics()

# 메타데이터 저장소 및 모델 캐시 초기화
metadata_store = {}
model_cache = OrderedDict()

# 모델 저장 경로 및 캐시 사이즈 설정
model_store_path = "../data/model_"
max_cache_size = 10  # 최대 캐시할 모델 개수

# 메트릭스 업데이트를 위한 함수
def load_model_to_cache(model_hash):
    """
    모델을 캐시에 로드하는 함수
    """
    try:
        if model_hash in model_cache:
            # 모델이 이미 캐시에 있는 경우, 순서 갱신
            model_cache.move_to_end(model_hash)
            logging.debug(f"Model {model_hash} accessed in cache.")
        else:
            if len(model_cache) >= max_cache_size:
                # 캐시가 꽉 찼을 경우, 가장 오래된 항목 제거
                oldest_key, _ = model_cache.popitem(last=False)
                logging.info(f"Removing oldest model from cache: {oldest_key}")
                metrics.set_model_cache_usage(len(model_cache))  # 캐시 업데이트

            # 모델 로드
            model_file_path = metadata_store[model_hash]['file_path']
            model = keras.models.load_model(model_file_path)
            model_cache[model_hash] = model
            logging.info(f"Model {model_hash} loaded into cache successfully.")
        
        # 캐시 사용량 업데이트
        metrics.set_model_cache_usage(len(model_cache))
        logging.debug(f"model_cache_usage set to {len(model_cache)}")
    except Exception as e:
        logging.error(f"Unexpected error occurred while loading model to cache: {str(e)}")
        metrics.increment_error_count('load_model_error')

@app.route('/metrics')
def metrics_endpoint():
    """
    Prometheus 메트릭스를 노출하는 엔드포인트
    """
    return Response(generate_latest(), mimetype='text/plain')

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
        metrics.increment_error_count('upload_model_missing_data')
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

        # 캐시 업데이트 (필요 시)
        load_model_to_cache(model_hash)

        response_message = 'File uploaded and processed successfully'
        logging.info(response_message)
        return jsonify({'message': response_message}), 200

    except Exception as e:
        metrics.increment_error_count('upload_model_error')
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
        response_message = 'Model hash is required'
        logging.error(response_message)
        metrics.increment_error_count('get_model_missing_hash')
        return jsonify({'error': response_message}), 400
    
    # 메타데이터 조회
    if model_hash not in metadata_store:
        response_message = 'No such model'
        logging.error(response_message)
        metrics.increment_error_count('get_model_not_found')
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
            response_message = 'Data and Model hash are required'
            logging.error(response_message)
            metrics.increment_error_count('predict_missing_data')
            return jsonify({'error': response_message}), 400
    
        # 캐시에서 모델 확인
        if model_hash in model_cache:
            metrics.increment_cache_hit()
            logging.debug(f"Cache hit for model {model_hash}")
        else:
            metrics.increment_cache_miss()
            logging.debug(f"Cache miss for model {model_hash}")
            load_model_to_cache(model_hash)

        model = model_cache.get(model_hash)
        if model is None:
            response_message = 'Model could not be loaded'
            logging.error(response_message)
            metrics.increment_error_count('predict_model_load_failed')
            return jsonify({'error': response_message}), 500

        # 모델 사용시간 업데이트
        metadata_store[model_hash]['used'] = tool_util.get_kr_time()

        # 데이터를 numpy 배열로 변환 및 예측
        array = np.array(data)
        prediction = model.predict(array)

        # 결과 완료 카운트
        metrics.increment_predictions_completed()
        logging.info(f"Prediction completed for model {model_hash}")

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        metrics.increment_error_count('predict')
        response_message = f'An error occurred: {str(e)}'
        logging.error(response_message)
        return jsonify({'error': response_message}), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Flask health check 
    """
    return "Healthy"

def load_metadata_store():
    """
    메타데이터 저장소 초기화
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
    오래된 모델 삭제
    """
    to_remove = []  # 제거할 항목의 키를 저장할 리스트
    for model_hash, metadata in metadata_store.items():
        if metadata['used'] < tool_util.one_week_ago():
            to_remove.append(model_hash)
    # 제거할 항목을 metadata_store에서 제거
    for model_hash in to_remove:
        remove_path = metadata_store[model_hash]['file_path']
        if os.path.exists(remove_path):
            shutil.rmtree(remove_path)
            logging.info(f"Removed old model: {model_hash}")
        del metadata_store[model_hash]
        metrics.set_model_cache_usage(len(model_cache))  # 캐시 업데이트

def sched_del_oldmodel():
    """
    스케줄링된 오래된 모델 삭제
    """
    threading.Timer(tool_util.delay_h(5), sched_del_oldmodel).start()
    del_oldmodel()
        
if __name__ == '__main__':

    log_level = os.getenv('LOG_LEVEL', 'DEBUG')  # 디버깅을 위해 DEBUG로 설정
    tool_util.set_logging(log_level)
    tool_util.set_folder()

    load_metadata_store()

    with ThreadPoolExecutor() as executor:
        executor.submit(sched_del_oldmodel)
        resource_monitor = ResourceMonitor()
        executor.submit(resource_monitor.start_monitor)

    app.run(host='0.0.0.0', port=5000)