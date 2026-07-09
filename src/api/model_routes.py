#api/model_routes
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from zipfile import BadZipFile
from src.common.metrics import get_metrics

model_bp = Blueprint('model', __name__)
metrics = get_metrics()


def _get_uploaded_file_size(model_file) -> int:
    stream = model_file.stream
    current_position = stream.tell()
    try:
        stream.seek(0, 2)
        return stream.tell()
    finally:
        stream.seek(current_position)


@model_bp.route('/upload_model', methods=['POST'])
def upload_model():
    """모델 업로드 엔드포인트"""
    model_file = request.files.get('model_file')
    model_hash = request.args.get('hash')

    # 필수 데이터 누락 확인
    if not model_file or not model_hash:
        metrics.increment_error_count('upload_model_missing_data')
        return jsonify({'error': 'Missing data (file or hash)'}), 400

    if _get_uploaded_file_size(model_file) > current_app.config['MAX_MODEL_FILE_SIZE']:
        metrics.increment_error_count('upload_model_too_large')
        return jsonify({'error': 'Uploaded file too large'}), 413

    try:
        # ModelManager를 통해 모델 저장 및 압축 해제
        msg, status = current_app.model_manager.upload_model(model_file, model_hash)
        return jsonify({'message': msg}), status

    except ValueError as e:
        metrics.increment_error_count('upload_model_error')
        return jsonify({'error': str(e)}), 400

    except BadZipFile:
        metrics.increment_error_count('upload_model_error')
        return jsonify({'error': 'Invalid zip file'}), 400

    except Exception as e:
        metrics.increment_error_count('upload_model_error')
        current_app.logger.error(f"Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@model_bp.route('/predict', methods=['POST'])
def predict():
    """예측 수행 엔드포인트"""
    model_hash = request.args.get('hash')
    data = request.get_json(silent=True)
    
    # 필수 파라미터 확인
    if not model_hash or not data:
        metrics.increment_error_count('predict_missing_data')
        return jsonify({'error': 'Missing hash or data'}), 400

    try:
        # 예측 수행 (ModelManager 위임)
        pred, status = current_app.model_manager.predict(model_hash, np.array(data))
        metrics.increment_predictions_completed()
        return jsonify({'prediction': pred.tolist()}), status

    except KeyError:
        metrics.increment_error_count('predict_model_not_found')
        return jsonify({'error': 'Model not found'}), 404

    except Exception as e:
        metrics.increment_error_count('predict_error')
        current_app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal error during prediction'}), 500

@model_bp.route('/get_model', methods=['GET'])
def get_model():
    """모델 존재 여부 및 정보 확인 엔드포인트"""
    model_hash = request.args.get('hash')
    
    if not model_hash:
        metrics.increment_error_count('get_model_missing_hash')  
        return jsonify({'error': 'Model hash is required'}), 400

    try:
        # 모델 정보 조회
        model_info = current_app.model_manager.get_model_info(model_hash)
        return jsonify({'message': model_info}), 200

    except KeyError:
        metrics.increment_error_count('get_model_not_found') 
        return jsonify({'error': 'No such model'}), 404

    except Exception as e:
        metrics.increment_error_count('get_model_error')  
        current_app.logger.error(f"Error retrieving model info: {e}")
        return jsonify({'error': str(e)}), 500
