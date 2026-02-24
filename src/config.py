#config
import os

class Config:
    """애플리케이션 설정 관리 클래스"""
    # 모델 저장 경로
    MODEL_STORE_PATH = os.getenv('MODEL_STORE_PATH', '../data/model_')
    
    # 로깅 설정
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

    # 로키 설정
    LOKI_URL = os.getenv('LOKI_URL')
    LOKI_TAGS = {
        "app": os.getenv('APP_NAME', 'ml-api'),
        "env": os.getenv('APP_ENV', 'dev')
    }
    
    # 모델 정리 주기 (시간)
    MODEL_CLEANUP_INTERVAL = int(os.getenv('MODEL_CLEANUP_INTERVAL', 5))
    
    # 서버 설정
    HOST = os.getenv('SERVER_HOST', '0.0.0.0')
    PORT = int(os.getenv('SERVER_PORT', 5000))
    
    # 기타 ML API 관련 설정
    MAX_MODEL_FILE_SIZE = int(os.getenv('MAX_MODEL_FILE_SIZE', 100 * 1024 * 1024))  # 100MB
