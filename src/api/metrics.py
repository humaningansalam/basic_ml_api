#api/metrics
from flask import Blueprint, Response
from prometheus_client import generate_latest

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics')
def metrics_endpoint():
    """Prometheus 메트릭스를 노출하는 엔드포인트"""
    # Prometheus 스크래핑을 위한 텍스트 형식 반환
    return Response(generate_latest(), mimetype='text/plain')