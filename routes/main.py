"""
Main web navigation routes blueprint.
Handles top-level dashboard, vehicle views, yard inventory, and analytics.
(Template rendering will be connected in Phase 2).
"""

from flask import Blueprint, jsonify

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Root landing route."""
    return jsonify({
        "message": "AI-Based Smart Yard Gate Automation System",
        "phase": "Phase 1 - Infrastructure Setup Complete",
        "status": "ready",
        "endpoints": {
            "health_check": "/api/health",
            "gate": "/gate",
            "dashboard": "/dashboard",
            "vehicles": "/vehicles",
            "yard": "/yard",
            "analytics": "/analytics",
        }
    })


@main_bp.route("/dashboard")
def dashboard():
    """Dashboard view route."""
    return jsonify({
        "module": "Dashboard",
        "description": "YMS Gate Automation Dashboard UI scheduled for Phase 2."
    })


@main_bp.route("/vehicles")
def vehicles():
    """Vehicles list route."""
    return jsonify({
        "module": "Vehicles",
        "description": "Vehicle log and details view scheduled for Phase 2."
    })


@main_bp.route("/yard")
def yard():
    """Yard inventory view route."""
    return jsonify({
        "module": "Yard Inventory",
        "description": "Yard location mapping and inventory view scheduled for Phase 11."
    })


@main_bp.route("/analytics")
def analytics():
    """Analytics view route."""
    return jsonify({
        "module": "Analytics",
        "description": "Gate KPIs and confidence analytics view scheduled for Phase 12."
    })
