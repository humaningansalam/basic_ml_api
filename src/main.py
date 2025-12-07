#main
import logging
import os
from flask import Flask
from src.config import Config
from src.api.health import health_bp
from src.api.metrics import metrics_bp
from src.api.model_routes import model_bp
from src.core.model_manager import ModelManager
from src.common.utils import set_folder
from src.common.metrics import get_metrics 
from his_mon import setup_logging, ResourceMonitor

def create_app(config_class=Config):
    """Flask 애플리케이션 팩토리 함수"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # 폴더 생성
    set_folder(app.config['MODEL_STORE_PATH'])
    
    # 모델 매니저 주입 (싱글톤처럼 앱 컨텍스트에 부착)
    app.model_manager = ModelManager(app.config['MODEL_STORE_PATH'])
    
    # 블루프린트 등록
    app.register_blueprint(health_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(model_bp)
    
    return app

if __name__ == '__main__':
    # 로깅 설정
    setup_logging(
        level=Config.LOG_LEVEL,
        loki_url=Config.LOKI_URL,
        tags=Config.LOKI_TAGS,
    )
    
    app = create_app()
    
    # 리소스 모니터 시작
    metrics = get_metrics()
    monitor = ResourceMonitor(metrics_obj=metrics, interval=5)
    monitor.start()
    
    app.run(host=Config.HOST, port=Config.PORT)