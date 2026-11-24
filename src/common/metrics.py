import threading
from typing import ClassVar

from his_mon import BaseMetrics
from prometheus_client import Counter, Gauge

from src.common.errors import ErrorCode


class ServiceMetrics(BaseMetrics):
    """Process-local Prometheus metrics used by the API and model manager."""

    _instance: ClassVar["ServiceMetrics | None"] = None
    _instance_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        super().__init__(app_name="ml_api")
        self._loaded_model_count = Gauge(
            "model_cache_usage",
            "Number of models currently cached",
        )
        self._completed_predictions = Counter(
            "predictions_completed",
            "Number of completed predictions",
        )
        self._cache_hits = Counter("cache_hits", "Number of cache hits")
        self._cache_misses = Counter("cache_misses", "Number of cache misses")

        # A labeled counter has no exported series until its first child exists.
        self.error_count.labels(type=ErrorCode.INTERNAL_ERROR.value)

    @classmethod
    def instance(cls) -> "ServiceMetrics":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def increment_error_count(self, error_code: ErrorCode) -> None:
        self.error_count.labels(type=error_code.value).inc()

    def increment_predictions_completed(self) -> None:
        self._completed_predictions.inc()

    def increment_cache_hit(self) -> None:
        self._cache_hits.inc()

    def increment_cache_miss(self) -> None:
        self._cache_misses.inc()

    def set_loaded_model_count(self, value: int) -> None:
        self._loaded_model_count.set(value)


def get_metrics() -> ServiceMetrics:
    return ServiceMetrics.instance()
