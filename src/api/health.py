from http import HTTPStatus

from flask import Blueprint, Response, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.get("/health")
def health_check() -> tuple[Response, int]:
    return jsonify({"data": {"status": "healthy"}}), HTTPStatus.OK
