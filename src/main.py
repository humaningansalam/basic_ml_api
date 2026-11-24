import threading

from flask import Flask
from his_mon import ResourceMonitor, setup_logging

from src.api.error_handlers import register_error_handlers
from src.api.health import health_bp
from src.api.metrics import metrics_bp
from src.api.model_routes import model_bp
from src.common.metrics import get_metrics
from src.config import Config
from src.core.model_manager import ModelManager

_REQUEST_OVERHEAD_BYTES = 1024 * 1024
_monitoring_lock = threading.Lock()
_resource_monitor: ResourceMonitor | None = None


def _start_resource_monitoring(app: Flask) -> ResourceMonitor | None:
    global _resource_monitor

    if not app.config.get("START_BACKGROUND_MONITORING", True):
        return None

    with _monitoring_lock:
        if _resource_monitor is None:
            setup_logging(
                level=app.config["LOG_LEVEL"],
                loki_url=app.config.get("LOKI_URL"),
                tags=app.config.get("LOKI_TAGS"),
            )
            _resource_monitor = ResourceMonitor(metrics_obj=get_metrics(), interval=5)
            _resource_monitor.start()

    return _resource_monitor


def create_app(config_class: type[Config] = Config) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_MODEL_FILE_SIZE"] + _REQUEST_OVERHEAD_BYTES

    app.extensions["model_manager"] = ModelManager(
        app.config["MODEL_STORE_PATH"],
        max_cached_models=app.config["MAX_CACHED_MODELS"],
        cleanup_interval_hours=app.config["MODEL_CLEANUP_INTERVAL_HOURS"],
        max_model_file_size=app.config["MAX_MODEL_FILE_SIZE"],
    )
    resource_monitor = _start_resource_monitoring(app)
    if resource_monitor is not None:
        app.extensions["resource_monitor"] = resource_monitor

    app.register_blueprint(health_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(model_bp)
    register_error_handlers(app)
    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host=application.config["HOST"], port=application.config["PORT"])
