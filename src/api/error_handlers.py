from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.common.errors import ApplicationError, ErrorCode
from src.common.metrics import get_metrics


@dataclass(frozen=True)
class ErrorSpec:
    status: Optional[HTTPStatus]
    message: str


@dataclass(frozen=True)
class ErrorResponse:
    code: ErrorCode
    message: str
    details: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error': {
                'code': self.code.value,
                'message': self.message,
                'details': dict(self.details),
            }
        }


ERROR_SPECS: Dict[ErrorCode, ErrorSpec] = {
    ErrorCode.MODEL_FILE_REQUIRED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'A model file is required.'),
    ErrorCode.MODEL_HASH_REQUIRED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'A model hash is required.'),
    ErrorCode.UPLOAD_TOO_LARGE: ErrorSpec(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, 'The uploaded model exceeds the size limit.'),
    ErrorCode.INVALID_MODEL_HASH: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The model hash format is invalid.'),
    ErrorCode.INVALID_ZIP: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The uploaded model is not a valid ZIP archive.'),
    ErrorCode.ZIP_ENTRY_LIMIT_EXCEEDED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The model archive contains too many entries.'),
    ErrorCode.UNSAFE_ZIP_ENTRY: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The model archive contains an unsafe path.'),
    ErrorCode.UNCOMPRESSED_SIZE_EXCEEDED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The expanded model archive exceeds the size limit.'),
    ErrorCode.MODEL_ARTIFACT_REQUIRED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The model archive must contain a .keras file.'),
    ErrorCode.MODEL_NOT_FOUND: ErrorSpec(HTTPStatus.NOT_FOUND, 'The requested model was not found.'),
    ErrorCode.MODEL_STORAGE_FAILED: ErrorSpec(HTTPStatus.INTERNAL_SERVER_ERROR, 'The model could not be stored.'),
    ErrorCode.MODEL_ARTIFACT_UNAVAILABLE: ErrorSpec(HTTPStatus.INTERNAL_SERVER_ERROR, 'The stored model artifact is unavailable.'),
    ErrorCode.PREDICTION_DATA_REQUIRED: ErrorSpec(HTTPStatus.BAD_REQUEST, 'Prediction data is required.'),
    ErrorCode.MALFORMED_JSON: ErrorSpec(HTTPStatus.BAD_REQUEST, 'The request body must contain valid JSON.'),
    ErrorCode.PREDICTION_DATA_NOT_ARRAY: ErrorSpec(HTTPStatus.BAD_REQUEST, 'Prediction data must be a JSON array.'),
    ErrorCode.PREDICTION_DATA_EMPTY: ErrorSpec(HTTPStatus.BAD_REQUEST, 'Prediction data must not be empty.'),
    ErrorCode.INVALID_PREDICTION_DATA: ErrorSpec(HTTPStatus.BAD_REQUEST, 'Prediction data has an invalid shape.'),
    ErrorCode.PREDICTION_FAILED: ErrorSpec(HTTPStatus.INTERNAL_SERVER_ERROR, 'The prediction could not be completed.'),
    ErrorCode.MODEL_STATE_CONFLICT: ErrorSpec(HTTPStatus.CONFLICT, 'The model changed while it was being loaded.'),
    ErrorCode.ROUTE_NOT_FOUND: ErrorSpec(HTTPStatus.NOT_FOUND, 'The requested API route was not found.'),
    ErrorCode.METHOD_NOT_ALLOWED: ErrorSpec(HTTPStatus.METHOD_NOT_ALLOWED, 'The HTTP method is not allowed for this route.'),
    ErrorCode.HTTP_ERROR: ErrorSpec(None, 'The HTTP request could not be completed.'),
    ErrorCode.INTERNAL_ERROR: ErrorSpec(HTTPStatus.INTERNAL_SERVER_ERROR, 'The request could not be completed.'),
}

HTTP_ERROR_CODES: Dict[int, ErrorCode] = {
    HTTPStatus.NOT_FOUND.value: ErrorCode.ROUTE_NOT_FOUND,
    HTTPStatus.METHOD_NOT_ALLOWED.value: ErrorCode.METHOD_NOT_ALLOWED,
}


def _error_response(error: ApplicationError, status_override: Optional[int] = None):
    spec = ERROR_SPECS[error.code]
    status = status_override if status_override is not None else spec.status
    if status is None:
        raise RuntimeError(f'HTTP status is required for error code {error.code.value}')
    response = ErrorResponse(error.code, spec.message, error.details)
    return jsonify(response.to_dict()), int(status)


def _copy_http_headers(response, error: HTTPException) -> None:
    for header, value in error.get_response().headers.items():
        if header.lower() not in {'content-type', 'content-length'}:
            response.headers[header] = value


def register_error_handlers(app: Flask) -> None:
    metrics = get_metrics()

    @app.errorhandler(ApplicationError)
    def handle_application_error(error: ApplicationError):
        metrics.increment_error_count(error.code)
        spec = ERROR_SPECS[error.code]
        if spec.status is not None and spec.status >= HTTPStatus.INTERNAL_SERVER_ERROR:
            app.logger.error(
                'Application error code=%s details=%s',
                error.code.value,
                dict(error.details),
                exc_info=(type(error), error, error.__traceback__),
            )
        return _error_response(error)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(error: RequestEntityTooLarge):
        content_length = request.content_length
        max_content_length = app.config.get('MAX_CONTENT_LENGTH')
        is_oversized_upload = (
            request.endpoint == 'model.upload_model'
            and content_length is not None
            and max_content_length is not None
            and content_length > max_content_length
        )

        if is_oversized_upload:
            application_error = ApplicationError(
                ErrorCode.UPLOAD_TOO_LARGE,
                {'max_bytes': app.config['MAX_MODEL_FILE_SIZE']},
            )
            status_override = None
        else:
            application_error = ApplicationError(
                ErrorCode.HTTP_ERROR,
                {'status': HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value},
            )
            status_override = HTTPStatus.REQUEST_ENTITY_TOO_LARGE.value

        metrics.increment_error_count(application_error.code)
        response, response_status = _error_response(
            application_error,
            status_override=status_override,
        )
        _copy_http_headers(response, error)
        return response, response_status

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        status = error.code or HTTPStatus.INTERNAL_SERVER_ERROR.value
        error_code = HTTP_ERROR_CODES.get(status, ErrorCode.HTTP_ERROR)
        details = {} if error_code is not ErrorCode.HTTP_ERROR else {'status': status}
        application_error = ApplicationError(error_code, details)
        metrics.increment_error_count(application_error.code)
        status_override = status if error_code is ErrorCode.HTTP_ERROR else None
        response, response_status = _error_response(
            application_error,
            status_override=status_override,
        )
        _copy_http_headers(response, error)
        return response, response_status

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        internal_error = ApplicationError(ErrorCode.INTERNAL_ERROR)
        metrics.increment_error_count(internal_error.code)
        app.logger.error(
            'Unhandled application error',
            exc_info=(type(error), error, error.__traceback__),
        )
        return _error_response(internal_error)
