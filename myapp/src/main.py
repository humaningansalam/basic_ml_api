import os
import logging
import numpy as np
from flask import Flask, request, jsonify, Response, current_app
from prometheus_client import generate_latest
from concurrent.futures import ThreadPoolExecutor
import threading

from myapp.src.model_manager import ModelManager
from myapp.src.monitor import ResourceMonitor
from myapp.common.prometheus_metric import get_metrics
import myapp.common.tool_util as tool_util

def create_app(model_store_path: str = "../data/model_") -> Flask:
    """Flask 애플리케이션 팩토리 함수"""
    app = Flask(__name__)
    model_manager = ModelManager(model_store_path)
    app.model_manager = model_manager

    metrics = get_metrics()  # 메트릭스 초기화

    @app.route('/metrics')
    def metrics_endpoint():
        """Prometheus 메트릭스를 노출하는 엔드포인트"""
        return Response(generate_latest(), mimetype='text/plain')

    @app.route('/upload_model', methods=['POST'])
    def upload_model():
        """모델 업로드 엔드포인트"""
        model_file = request.files.get('model_file')
        model_hash = request.args.get('hash')

        if not model_file or not model_hash:
            metrics.increment_error_count('upload_model_missing_data')  
            return jsonify({'error': 'Model file and hash are required'}), 400

        try:
            message, status_code = current_app.model_manager.upload_model(model_file, model_hash)
            return jsonify({'message': message}), status_code
        except Exception as e:
            metrics.increment_error_count('upload_model_error') 
            logging.error(f"Error uploading model: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/get_model', methods=['GET'])
    def get_model():
        """모델 존재 확인 엔드포인트"""
        model_hash = request.args.get('hash')
        if not model_hash:
            metrics.increment_error_count('get_model_missing_hash')  
            return jsonify({'error': 'Model hash is required'}), 400

        try:
            model_info = current_app.model_manager.get_model_info(model_hash)
            return jsonify({'message': model_info}), 200
        except KeyError:
            metrics.increment_error_count('get_model_not_found') 
            return jsonify({'error': 'No such model'}), 404
        except Exception as e:
            metrics.increment_error_count('get_model_error')  
            logging.error(f"Error retrieving model info: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict():
        """예측 수행 엔드포인트"""
        model_hash = request.args.get('hash')
        if not model_hash:
            metrics.increment_error_count('predict_missing_hash') 
            return jsonify({'error': 'Model hash is required'}), 400

        data = request.get_json()
        if not data:
            metrics.increment_error_count('predict_missing_data')  
            return jsonify({'error': 'Data is required'}), 400

        try:
            prediction, status_code = current_app.model_manager.predict(model_hash, np.array(data))
            metrics.increment_predictions_completed()
            return jsonify({'prediction': prediction.tolist()}), status_code
        except KeyError:
            metrics.increment_error_count('predict_model_not_found')  
            return jsonify({'error': 'Model not found'}), 404
        except OSError:
            metrics.increment_error_count('predict_model_file_not_found') 
            return jsonify({'error': 'Model file not found'}), 500
        except Exception as e:
            metrics.increment_error_count('predict_error')  
            logging.error(f"Unexpected error during prediction: {e}")
            return jsonify({'error': 'An error occurred during prediction'}), 500

    @app.route('/health', methods=['GET'])
    def health():
        """헬스체크 엔드포인트"""
        return "Healthy"

    def start_cleanup_scheduler():
        """주기적인 모델 정리 스케줄러"""
        def scheduled_cleanup():
            while True:
                current_app.model_manager.clean_old_models()
                tool_util.delay_h(5)  # 5시간 대기

        cleanup_thread = threading.Thread(target=scheduled_cleanup, daemon=True)
        cleanup_thread.start()

    # 리소스 모니터링 및 정리 스케줄러 시작
    with ThreadPoolExecutor() as executor:
        executor.submit(start_cleanup_scheduler)
        resource_monitor = ResourceMonitor()
        executor.submit(resource_monitor.start_monitor)

    return app

if __name__ == '__main__':
    log_level = os.getenv('LOG_LEVEL', 'DEBUG')
    tool_util.set_logging(log_level)
    tool_util.set_folder()

    app = create_app()
    app.run(host='0.0.0.0', port=5000)