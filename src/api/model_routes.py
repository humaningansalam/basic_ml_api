import json
from http import HTTPStatus
from typing import cast

import numpy as np
from flask import Blueprint, Response, current_app, jsonify, request
from werkzeug.datastructures import FileStorage

from src.common.errors import ApplicationError, ErrorCode
from src.common.metrics import get_metrics
from src.core.model_manager import ModelManager

model_bp = Blueprint("model", __name__)


def _get_model_manager() -> ModelManager:
    return cast(ModelManager, current_app.extensions["model_manager"])


def _reject_non_standard_json_number(value: str) -> None:
    raise ValueError(f"Non-standard JSON numeric token: {value}")


def _uploaded_file_size(model_file: FileStorage) -> int:
    stream = model_file.stream
    current_position = stream.tell()
    try:
        stream.seek(0, 2)
        return stream.tell()
    finally:
        stream.seek(current_position)


def _require_model_hash() -> str:
    model_hash = request.args.get("hash")
    if not model_hash:
        raise ApplicationError(ErrorCode.MODEL_HASH_REQUIRED)
    return model_hash


def _parse_prediction_input() -> np.ndarray:
    if not request.is_json:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_REQUIRED)

    raw_body = request.get_data(cache=True)
    if not raw_body:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_REQUIRED)

    try:
        parsed_body = json.loads(
            raw_body,
            parse_constant=_reject_non_standard_json_number,
        )
    except RecursionError as error:
        raise ApplicationError(ErrorCode.INVALID_PREDICTION_DATA) from error
    except (UnicodeDecodeError, ValueError) as error:
        raise ApplicationError(ErrorCode.MALFORMED_JSON) from error

    if not isinstance(parsed_body, list):
        raise ApplicationError(ErrorCode.PREDICTION_DATA_NOT_ARRAY)
    if not parsed_body:
        raise ApplicationError(ErrorCode.PREDICTION_DATA_EMPTY)

    try:
        return np.asarray(parsed_body)
    except ValueError as error:
        raise ApplicationError(ErrorCode.INVALID_PREDICTION_DATA) from error


@model_bp.post("/upload_model")
def upload_model() -> tuple[Response, int]:
    model_file = request.files.get("model_file")
    if model_file is None or not model_file.filename:
        raise ApplicationError(ErrorCode.MODEL_FILE_REQUIRED)

    model_hash = _require_model_hash()
    max_file_size = current_app.config["MAX_MODEL_FILE_SIZE"]
    if _uploaded_file_size(model_file) > max_file_size:
        raise ApplicationError(ErrorCode.UPLOAD_TOO_LARGE, {"max_bytes": max_file_size})

    upload_result = _get_model_manager().upload_model(model_file, model_hash)
    return jsonify({"data": upload_result.to_dict()}), HTTPStatus.OK


@model_bp.post("/predict")
def predict_with_model() -> tuple[Response, int]:
    model_hash = _require_model_hash()
    prediction_input = _parse_prediction_input()
    prediction_result = _get_model_manager().predict(model_hash, prediction_input)
    get_metrics().increment_predictions_completed()
    return jsonify({"data": prediction_result.to_dict()}), HTTPStatus.OK


@model_bp.get("/get_model")
def get_model_info() -> tuple[Response, int]:
    model_hash = _require_model_hash()
    model_info = _get_model_manager().get_model_info(model_hash)
    return jsonify({"data": model_info.to_dict()}), HTTPStatus.OK
