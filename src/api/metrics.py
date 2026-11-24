from flask import Blueprint, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.get("/metrics")
def metrics_endpoint() -> Response:
    return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)
