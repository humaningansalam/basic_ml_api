#main
import logging
from flask import Flask
from src.config import Config
from src.api.health import health_bp
from src.api.metrics import metrics_bp
from src.api.model_routes import model_bp
from src.core.model_manager import ModelManager
from src.core.monitor import ResourceMonitor
from src.common.utils import set_folder

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
    logging.basicConfig(level=Config.LOG_LEVEL)
    
    app = create_app()
    
    # 리소스 모니터 시작
    monitor = ResourceMonitor()
    monitor.start_monitor()
    
    app.run(host=Config.HOST, port=Config.PORT)