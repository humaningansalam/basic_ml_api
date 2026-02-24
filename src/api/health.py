#api/health
from flask import Blueprint

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """서비스 헬스 체크 엔드포인트 (Liveness Probe용)"""
    return "Healthy", 200