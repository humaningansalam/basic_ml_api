#main
import logging
import os
from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge
from src.config import Config
from src.api.health import health_bp
from src.api.metrics import metrics_bp
from src.api.model_routes import model_bp
from src.core.model_manager import ModelManager
from src.common.utils import set_folder
from src.common.metrics import get_metrics
from his_mon import setup_logging, ResourceMonitor

_setup_done = False


def _should_start_monitoring(config_class) -> bool:
    return getattr(config_class, 'START_BACKGROUND_MONITORING', True)


def create_app(config_class=Config):
    """Flask 애플리케이션 팩토리 함수"""
    app = Flask(__name__)
    app.config.from_object(config_class)

    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_MODEL_FILE_SIZE"]

    set_folder(app.config['MODEL_STORE_PATH'])

    app.model_manager = ModelManager(
        app.config['MODEL_STORE_PATH'],
        cleanup_interval_hours=app.config['MODEL_CLEANUP_INTERVAL'],
        max_model_file_size=app.config['MAX_MODEL_FILE_SIZE'],
    )

    if _should_start_monitoring(config_class) and not _setup_done:
        setup_logging(
            level=app.config["LOG_LEVEL"],
            loki_url=app.config.get("LOKI_URL"),
            tags=app.config.get("LOKI_TAGS"),
        )

        metrics = get_metrics()
        monitor = ResourceMonitor(metrics_obj=metrics, interval=5)
        monitor.start()

        globals()['_setup_done'] = True

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_entity_too_large(error):
        return jsonify({'error': 'Uploaded file too large'}), 413

    app.register_blueprint(health_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(model_bp)

    return app

if __name__ == '__main__':
    app = create_app()

    app.run(host=Config.HOST, port=Config.PORT)
