"""
Gate automation routes blueprint.
Handles gate camera feed, vehicle detection workflow, and check-in/out UI.
"""

from flask import Blueprint, jsonify

gate_bp = Blueprint("gate", __name__, url_prefix="/gate")


@gate_bp.route("/")
def gate_index():
    """Gate automation landing route."""
    return jsonify({
        "module": "Gate Automation",
        "description": "Interactive gate upload and detection view scheduled for Phase 2 UI integration."
    })
