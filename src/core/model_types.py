import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import tensorflow as tf


def _to_json_value(value: Any) -> Any:
    if tf.is_tensor(value):
        return _to_json_value(value.numpy())
    if isinstance(value, np.ndarray):
        return _to_json_value(value.tolist())
    if isinstance(value, np.generic):
        return _to_json_value(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Prediction output mappings must use string keys")
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Prediction output contains non-finite values")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported prediction output type: {type(value).__name__}")


@dataclass(slots=True)
class ModelMetadata:
    file_path: str
    last_used_at: datetime


@dataclass(frozen=True, slots=True)
class ModelInfo:
    model_hash: str
    file_path: str
    last_used_at: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "model_hash": self.model_hash,
            "file_path": self.file_path,
            "used": self.last_used_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class UploadResult:
    model_hash: str
    replaced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "replaced": self.replaced,
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    model_hash: str
    values: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _to_json_value(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_hash": self.model_hash,
            "prediction": self.values,
        }


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed_model_hashes: tuple[str, ...]
