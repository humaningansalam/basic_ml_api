#api/health
from enum import StrEnum

from flask import Blueprint, jsonify


class HealthStatus(StrEnum):
    HEALTHY = 'healthy'

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    """서비스 헬스 체크 엔드포인트 (Liveness Probe용)"""
    return jsonify({'data': {'status': HealthStatus.HEALTHY.value}}), 200
