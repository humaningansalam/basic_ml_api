import os
from typing import ClassVar


def _read_int_setting(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error

    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{name} must be at most {maximum}")
    return value


class Config:
    """Default Flask configuration loaded from environment variables."""

    MODEL_STORE_PATH = os.getenv("MODEL_STORE_PATH", "data/model_")
    MAX_MODEL_FILE_SIZE = _read_int_setting(
        "MAX_MODEL_FILE_SIZE",
        100 * 1024 * 1024,
        minimum=1,
    )
    MAX_CACHED_MODELS = _read_int_setting(
        "MAX_CACHED_MODELS",
        10,
        minimum=1,
    )
    MODEL_CLEANUP_INTERVAL_HOURS = _read_int_setting(
        "MODEL_CLEANUP_INTERVAL",
        5,
        minimum=0,
    )

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOKI_URL = os.getenv("LOKI_URL")
    LOKI_TAGS: ClassVar[dict[str, str]] = {
        "app": os.getenv("APP_NAME", "ml-api"),
        "env": os.getenv("APP_ENV", "dev"),
    }
    START_BACKGROUND_MONITORING = True

    HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    PORT = _read_int_setting("SERVER_PORT", 5000, minimum=1, maximum=65535)
