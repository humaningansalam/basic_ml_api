#common/metrics
from prometheus_client import Counter, Gauge
import threading

class PMetrics:
    _lock = threading.Lock()
    _instance = None

    def __init__(self):
        # MLApi metrics
        self.model_cache_usage = Gauge('model_cache_usage', 'Number of models currently cached')
        self.errors_count = Counter('errors', 'Number of errors', ['type'])
        self.predictions_completed = Counter('predictions_completed', 'Number of completed predictions')
        self.cache_hits = Counter('cache_hits', 'Number of cache hits')
        self.cache_misses = Counter('cache_misses', 'Number of cache misses')

        # Monitor metrics
        self.app_cpu_usage = Gauge('app_cpu_usage', 'CPU usage of the application')
        self.app_ram_usage = Gauge('app_ram_usage', 'RAM usage of the application')

    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 반환"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = PMetrics()
        return cls._instance

    # Metric 메서드들
    def increment_error_count(self, error_type):
        self.errors_count.labels(type=error_type).inc()

    def increment_predictions_completed(self):
        self.predictions_completed.inc()
        
    def set_app_cpu_usage(self, value):
        self.app_cpu_usage.set(value)

    def set_app_ram_usage(self, value):
        self.app_ram_usage.set(value)

    def increment_cache_hit(self):
        self.cache_hits.inc()

    def increment_cache_miss(self):
        self.cache_misses.inc()

    def set_model_cache_usage(self, value):
        self.model_cache_usage.set(value)

def get_metrics():
    """메트릭스 인스턴스 획득 헬퍼 함수"""
    return PMetrics.get_instance()