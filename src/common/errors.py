from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Any, Mapping


@unique
class ErrorCode(StrEnum):
    MODEL_FILE_REQUIRED = 'model_file_required'
    MODEL_HASH_REQUIRED = 'model_hash_required'
    UPLOAD_TOO_LARGE = 'upload_too_large'
    INVALID_MODEL_HASH = 'invalid_model_hash'
    INVALID_ZIP = 'invalid_zip'
    ZIP_ENTRY_LIMIT_EXCEEDED = 'zip_entry_limit_exceeded'
    UNSAFE_ZIP_ENTRY = 'unsafe_zip_entry'
    UNCOMPRESSED_SIZE_EXCEEDED = 'uncompressed_size_exceeded'
    MODEL_ARTIFACT_REQUIRED = 'model_artifact_required'
    MODEL_NOT_FOUND = 'model_not_found'
    MODEL_STORAGE_FAILED = 'model_storage_failed'
    MODEL_ARTIFACT_UNAVAILABLE = 'model_artifact_unavailable'
    PREDICTION_DATA_REQUIRED = 'prediction_data_required'
    MALFORMED_JSON = 'malformed_json'
    PREDICTION_DATA_NOT_ARRAY = 'prediction_data_not_array'
    PREDICTION_DATA_EMPTY = 'prediction_data_empty'
    INVALID_PREDICTION_DATA = 'invalid_prediction_data'
    PREDICTION_FAILED = 'prediction_failed'
    MODEL_STATE_CONFLICT = 'model_state_conflict'
    ROUTE_NOT_FOUND = 'route_not_found'
    METHOD_NOT_ALLOWED = 'method_not_allowed'
    HTTP_ERROR = 'http_error'
    INTERNAL_ERROR = 'internal_error'


@dataclass(eq=False)
class ApplicationError(Exception):
    code: ErrorCode
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.code.value)
