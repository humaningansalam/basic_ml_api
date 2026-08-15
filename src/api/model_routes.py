import json

import numpy as np
from flask import Blueprint, current_app, jsonify, request

from src.common.errors import ApplicationError, ErrorCode
from src.common.metrics import get_metrics


model_bp = Blueprint('model', __name__)
metrics = get_metrics()


def _reject_non_standard_json_number(value: str) -> None:
    raise ValueError(f'Non-standard JSON numeric token: {value}')


def _get_uploaded_file_size(model_file) -> int:
    stream = model_file.stream
    current_position = stream.tell()
    try:
        stream.seek(0, 2)
        return stream.tell()
    finally:
        stream.seek(current_position)


def _require_model_hash() -> str:
    model_hash = request.args.get('hash')
    if not model_hash:
        raise ApplicationError(ErrorCode.MODEL_HASH_REQUIRED)
    return model_hash


def _parse_prediction_data() -> np.ndarray:
    if not request.is_json:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_REQUIRED)
    payload = request.get_data(cache=True)
    if not payload:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_REQUIRED)

    try:
        data = json.loads(
            payload,
            parse_constant=_reject_non_standard_json_number,
        )
    except RecursionError as error:
        raise ApplicationError(ErrorCode.INVALID_PREDICTION_DATA) from error
    except (UnicodeDecodeError, ValueError) as error:
        raise ApplicationError(ErrorCode.MALFORMED_JSON) from error

    if not isinstance(data, list):
        raise ApplicationError(ErrorCode.PREDICTION_DATA_NOT_ARRAY)
    if not data:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_EMPTY)

    try:
        return np.asarray(data)
    except ValueError as error:
        raise ApplicationError(ErrorCode.INVALID_PREDICTION_DATA) from error


@model_bp.route('/upload_model', methods=['POST'])
def upload_model():
    model_file = request.files.get('model_file')
    if model_file is None or not model_file.filename:
        raise ApplicationError(ErrorCode.MODEL_FILE_REQUIRED)
    model_hash = _require_model_hash()

    max_file_size = current_app.config['MAX_MODEL_FILE_SIZE']
    if _get_uploaded_file_size(model_file) > max_file_size:
        raise ApplicationError(ErrorCode.UPLOAD_TOO_LARGE, {'max_bytes': max_file_size})

    result = current_app.model_manager.upload_model(model_file, model_hash)
    return jsonify({'data': result.to_dict()}), 200


@model_bp.route('/predict', methods=['POST'])
def predict():
    model_hash = _require_model_hash()
    data = _parse_prediction_data()
    result = current_app.model_manager.predict(model_hash, data)
    metrics.increment_predictions_completed()
    return jsonify({'data': result.to_dict()}), 200


@model_bp.route('/get_model', methods=['GET'])
def get_model():
    model_hash = _require_model_hash()
    model_info = current_app.model_manager.get_model_info(model_hash)
    return jsonify({'data': model_info.to_dict()}), 200
