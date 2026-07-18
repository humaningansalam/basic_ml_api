from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import tensorflow as tf


def _normalize_prediction_output(value: Any) -> Any:
    if tf.is_tensor(value):
        return _normalize_prediction_output(value.numpy())
    if isinstance(value, np.ndarray):
        return _normalize_prediction_output(value.tolist())
    if isinstance(value, np.generic):
        return _normalize_prediction_output(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('Prediction output mappings must use string keys')
        return {key: _normalize_prediction_output(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_prediction_output(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Prediction output contains non-finite values')
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f'Unsupported prediction output type: {type(value).__name__}')


@dataclass
class ModelMetadata:
    file_path: str
    used: datetime


@dataclass(frozen=True)
class ModelInfo:
    model_hash: str
    file_path: str
    used: datetime

    def to_dict(self) -> Dict[str, str]:
        return {
            'model_hash': self.model_hash,
            'file_path': self.file_path,
            'used': self.used.isoformat(),
        }


@dataclass(frozen=True)
class UploadResult:
    model_hash: str
    replaced: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_hash': self.model_hash,
            'replaced': self.replaced,
        }


@dataclass(frozen=True)
class PredictionResult:
    model_hash: str
    values: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', _normalize_prediction_output(self.values))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_hash': self.model_hash,
            'prediction': self.values,
        }


@dataclass(frozen=True)
class CleanupResult:
    removed_model_hashes: Tuple[str, ...]
