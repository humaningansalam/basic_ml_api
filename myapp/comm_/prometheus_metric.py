from prometheus_client import Gauge, Counter

class PMetrics:
    def __init__(self):
        # MLApi metrics
        self.model_cache_usage = Gauge('model_cache_usage', 'Number of models currently cached')
        self.errors_count = Counter('errors', 'Number of errors', ['type'])
        self.predictions_completed = Counter('predictions_completed', 'Number of completed predictions')

        # Monitor metrics
        self.app_cpu_usage = Gauge('app_cpu_usage', 'CPU usage of the application')
        self.app_ram_usage = Gauge('app_ram_usage', 'RAM usage of the application')

    # MLApi metric methods
    def set_model_cache_usage(self, value):
        self.model_cache_usage.set(value)

    def increment_error_count(self, error_type):
        self.errors_count.labels(type=error_type).inc()

    def increment_predictions_completed(self):
        self.predictions_completed.inc()

    # Monitor metric methods
    def set_app_cpu_usage(self, value):
        self.app_cpu_usage.set(value)

    def set_app_ram_usage(self, value):
        self.app_ram_usage.set(value)

_instance = None

def get_metrics():
    global _instance
    if _instance is None:
        _instance = PMetrics()
    return _instance